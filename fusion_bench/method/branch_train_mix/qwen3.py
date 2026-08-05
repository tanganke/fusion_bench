from dataclasses import dataclass
from typing import cast

import torch
from transformers import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeExperts,
    Qwen3MoeMLP,
)

from fusion_bench.method.simple_average import simple_average


@dataclass
class Qwen3MoEArgs:
    """
    Args:

        num_experts: Total number of routed experts.
        num_experts_per_tok: Number of experts activated per token.
        moe_intermediate_size: Intermediate size of each routed expert's MLP.
        decoder_sparse_step: Frequency of MoE layers (every N layers is a MoE layer).
        norm_topk_prob: Whether to normalize the top-k routing probabilities.
        output_router_logits: Whether to return router logits (enables auxiliary loss).
        router_aux_loss_coef: Coefficient for the load-balancing auxiliary loss.
        mlp_only_layers: Layer indices that use a plain MLP instead of a MoE block.
            If ``None``, defaults to an empty list (use ``decoder_sparse_step`` to
            determine sparsity).
    """

    num_experts: int
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 768
    decoder_sparse_step: int = 1
    norm_topk_prob: bool = False
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    mlp_only_layers: list[int] | None = None


def construct_qwen3_moe_config_from_base(
    base_config: Qwen3Config,
    moe_args: Qwen3MoEArgs,
) -> Qwen3MoeConfig:
    """Constructs a Qwen3MoeConfig from a base Qwen3Config by copying relevant parameters and setting the number of experts.

    Args:
        base_config: The base Qwen3Config to copy parameters from.

    """
    return Qwen3MoeConfig(
        vocab_size=base_config.vocab_size,
        hidden_size=base_config.hidden_size,
        intermediate_size=base_config.intermediate_size,
        num_hidden_layers=base_config.num_hidden_layers,
        num_attention_heads=base_config.num_attention_heads,
        num_key_value_heads=base_config.num_key_value_heads,
        hidden_act=base_config.hidden_act,
        max_position_embeddings=base_config.max_position_embeddings,
        initializer_range=base_config.initializer_range,
        rms_norm_eps=base_config.rms_norm_eps,
        use_cache=base_config.use_cache,
        tie_word_embeddings=base_config.tie_word_embeddings,
        rope_parameters=base_config.rope_parameters,
        attention_bias=base_config.attention_bias,
        use_sliding_window=base_config.use_sliding_window,
        sliding_window=base_config.sliding_window,
        attention_dropout=base_config.attention_dropout,
        # MoE arguments
        num_experts=moe_args.num_experts,
        num_experts_per_tok=moe_args.num_experts_per_tok,
        moe_intermediate_size=moe_args.moe_intermediate_size,
        decoder_sparse_step=moe_args.decoder_sparse_step,
        norm_topk_prob=moe_args.norm_topk_prob,
        output_router_logits=moe_args.output_router_logits,
        router_aux_loss_coef=moe_args.router_aux_loss_coef,
        mlp_only_layers=moe_args.mlp_only_layers,
    )


def mix_qwen3_models_to_moe(
    base_model: Qwen3ForCausalLM,
    expert_models: list[Qwen3ForCausalLM],
    **kwargs,
) -> Qwen3MoeForCausalLM:
    """Mixes the parameters of a base Qwen3 model and multiple expert Qwen3 models into a single Qwen3MoeForCausalLM model.

    Args:
        base_model: The base Qwen3 model to use as the foundation for the MoE model.
        expert_models: A list of expert Qwen3 models whose parameters will be mixed into the MoE model.

    Returns:
        A Qwen3MoeForCausalLM model with parameters mixed from the base and expert models.
    """
    moe_args = Qwen3MoEArgs(num_experts=len(expert_models))

    # Construct the MoE config from the base model's config and the provided MoE arguments
    moe_config = construct_qwen3_moe_config_from_base(base_model.config, moe_args)

    # Initialize a new MoE model with the constructed config
    moe_model = Qwen3MoeForCausalLM(moe_config)

    # Average the parameters of the expert models into the MoE model's expert submodules
    def _average_expert_parameters(module_name):
        print(f"Averaging parameters for module: {module_name}")
        base_module = moe_model.get_submodule(module_name)
        expert_modules = [
            exp_model.get_submodule(module_name) for exp_model in expert_models
        ]
        simple_average(
            modules=expert_modules,
            base_module=base_module,
        )

    for module_name in [
        "lm_head",
        "model.embed_tokens",
        "model.norm",
        "model.rotary_emb",
    ]:
        _average_expert_parameters(module_name)

    for layer_idx in range(moe_config.num_hidden_layers):
        _average_expert_parameters(f"model.layers.{layer_idx}.self_attn")
        _average_expert_parameters(f"model.layers.{layer_idx}.input_layernorm")
        _average_expert_parameters(f"model.layers.{layer_idx}.post_attention_layernorm")

        if isinstance(moe_model.model.layers[layer_idx].mlp, Qwen3MoeMLP):
            _average_expert_parameters(f"model.layers.{layer_idx}.mlp")
        elif isinstance(moe_model.model.layers[layer_idx].mlp, Qwen3MoeExperts):
            moe_mlp = cast(Qwen3MoeExperts, moe_model.model.layers[layer_idx].mlp)
            expert_mlps = [
                exp_model.model.layers[layer_idx].mlp for exp_model in expert_models
            ]

            # gate, up proj
            gate_up_expert_weights = torch.stack(
                [
                    torch.cat(
                        [
                            expert_mlp.gate_proj.weight,
                            expert_mlp.up_proj.weight,
                        ],
                        dim=0,
                    )
                    for expert_mlp in expert_mlps
                ],
                dim=0,
            )
            moe_mlp.gate_up_proj.weight.data.copy_(gate_up_expert_weights)

            # down proj
            down_proj_expert_weights = torch.stack(
                [expert_mlp.down_proj.weight for expert_mlp in expert_mlps], dim=0
            )
            moe_mlp.down_proj.weight.data.copy_(down_proj_expert_weights)

    return moe_model
