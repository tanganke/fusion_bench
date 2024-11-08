"""
This module contains the implementation of ExPO merge for LLAMA models.

Reference:
- Zheng et al. Weak-to-Strong Extrapolation Expedites Alignment.
"""

import logging
from typing import cast

from torch import nn
from transformers import LlamaForCausalLM, LlamaModel

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.method import SimpleAverageAlgorithm
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
)

log = logging.getLogger(__name__)


def expo_(sft_model: nn.Module, rlhf_model: nn.Module, extrapolation_factor: float):
    """
    Applies extrapolation to the parameters of the RLHF model based on the SFT model.
    The RLHF model is updated in place.

    Args:
        sft_model (nn.Module): The supervised fine-tuned model.
        rlhf_model (nn.Module): The reinforcement learning from human feedback model.
        extrapolation_factor (float): The factor by which to extrapolate the parameters.

    Returns:
        nn.Module: The RLHF model with updated parameters.
    """
    delta_parameters = state_dict_sub(rlhf_model.state_dict(), sft_model.state_dict())
    merged_sd = state_dict_add(
        rlhf_model.state_dict(),
        state_dict_mul(delta_parameters, scalar=extrapolation_factor),
    )

    rlhf_model.load_state_dict(merged_sd)
    return rlhf_model


def expo_linear_modules_(
    sft_model: nn.Module, rlhf_model: nn.Module, extrapolation_factor: float
):
    """
    Applies extrapolation to the linear modules of the RLHF model based on the SFT model.
    The RLHF model is updated in place.

    Args:
        sft_model (nn.Module): The supervised fine-tuned model.
        rlhf_model (nn.Module): The reinforcement learning from human feedback model.
        extrapolation_factor (float): The factor by which to extrapolate the parameters.

    Returns:
        nn.Module: The RLHF model with updated linear modules.
    """
    for name, module in sft_model.named_modules():
        if isinstance(module, nn.Linear):
            expo_(module, rlhf_model.get_submodule(name), extrapolation_factor)
    return rlhf_model


class ExPOAlgorithmForLlama(BaseAlgorithm):

    def __init__(
        self,
        extrapolation_factor: float,
        attention_scaling_factor: float = 0.5,
        only_on_backbone: bool = True,
        on_linear_weights: bool = True,
        on_linear_bias: bool = False,
        on_embedding: bool = False,
        **kwargs,
    ):
        self.extrapolation_factor = extrapolation_factor
        self.attention_scaling_factor = attention_scaling_factor
        self.only_on_backbone = only_on_backbone
        self.on_linear_weights = on_linear_weights
        self.on_linear_bias = on_linear_bias
        self.on_embedding = on_embedding
        super().__init__(**kwargs)

    def run(self, modelpool: BaseModelPool):
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        assert len(modelpool.model_names) >= 1, "ExPO requires at least one model."
        assert modelpool.has_pretrained, "ExPO requires pretrained models (base model)."

        sft_model: LlamaForCausalLM = modelpool.load_pretrained_model()
        if len(modelpool) == 1:
            rlhf_model = modelpool.load_model(modelpool.model_names[0])
        else:
            # if there are multiple RLHF models, use simple average to merge them before running ExPO
            log.info(
                f"There are {len(modelpool)} models in the model pool, averaging them first..."
            )
            rlhf_model = SimpleAverageAlgorithm().run(modelpool)
        rlhf_model = cast(LlamaForCausalLM, rlhf_model)

        if not self.on_linear_bias:
            for name, module in sft_model.named_modules():
                if isinstance(module, nn.Linear):
                    module.bias = rlhf_model.get_submodule(name).bias
        if not self.on_linear_weights:
            for name, module in sft_model.named_modules():
                if isinstance(module, nn.Linear):
                    module.weight = rlhf_model.get_submodule(name).weight

        if not self.only_on_backbone:
            expo_(sft_model.lm_head, rlhf_model.lm_head, self.extrapolation_factor)

        # expo on the backbone
        self._expo_lm_model_(
            sft_model.model, rlhf_model.model, self.extrapolation_factor
        )
        return rlhf_model

    def _expo_lm_model_(
        self, sft_model: LlamaModel, rlhf_model: LlamaModel, extrapolation_factor
    ):
        if self.on_embedding:
            expo_(sft_model.embed_tokens, rlhf_model.embed_tokens, extrapolation_factor)
        for layer_idx, layer in enumerate(sft_model.layers):
            expo_linear_modules_(
                layer.self_attn,
                rlhf_model.layers[layer_idx].self_attn,
                extrapolation_factor * self.attention_scaling_factor,
            )
            expo_linear_modules_(
                layer.mlp,
                rlhf_model.layers[layer_idx].mlp,
                extrapolation_factor,
            )
