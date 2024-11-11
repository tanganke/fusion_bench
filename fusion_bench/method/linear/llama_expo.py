"""
This module contains the implementation of ExPO merge for LLAMA models.

Reference:
- Zheng et al. Weak-to-Strong Extrapolation Expedites Alignment.
"""

import logging
from typing import cast, Optional

import torch
from torch import nn
from transformers import LlamaForCausalLM, LlamaModel

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.method import SimpleAverageAlgorithm
from fusion_bench.utils.state_dict_arithmetic import StateDictType

log = logging.getLogger(__name__)


def expo_(
    sft_model: nn.Module,
    rlhf_model: nn.Module,
    extrapolation_factor: float,
    merge_dtype: Optional[torch.dtype] = None,
):
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
    rlhf_state_dict: StateDictType = rlhf_model.state_dict()
    sft_state_dict: StateDictType = sft_model.state_dict()

    merged_state_dict = {}

    for n in rlhf_state_dict:
        rlhf_p = rlhf_state_dict[n]
        sft_p = sft_state_dict[n]
        if merge_dtype is not None:
            orignal_dtype = rlhf_state_dict[n].dtype
            rlhf_p = rlhf_state_dict[n].to(dtype=merge_dtype)
            sft_p = sft_state_dict[n].to(dtype=merge_dtype)

        rlhf_state_dict[n] = rlhf_p + extrapolation_factor * (rlhf_p - sft_p)

        if merge_dtype is not None:
            merged_state_dict[n] = rlhf_p.to(dtype=orignal_dtype)
        else:
            merged_state_dict[n] = rlhf_p

    rlhf_model.load_state_dict(merged_state_dict)
    return rlhf_model


def expo_linear_modules_(
    sft_model: nn.Module,
    rlhf_model: nn.Module,
    extrapolation_factor: float,
    merge_dtype: Optional[torch.dtype] = None,
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
            expo_(
                module,
                rlhf_model.get_submodule(name),
                extrapolation_factor=extrapolation_factor,
                merge_dtype=merge_dtype,
            )
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
        fix_last_n_layers: int = 0,
        fix_first_n_layers: int = 0,
        **kwargs,
    ):
        self.extrapolation_factor = extrapolation_factor
        self.attention_scaling_factor = attention_scaling_factor
        self.only_on_backbone = only_on_backbone
        self.on_linear_weights = on_linear_weights
        self.on_linear_bias = on_linear_bias
        self.on_embedding = on_embedding
        self.fix_last_n_layers = fix_last_n_layers
        self.fix_first_n_layers = fix_first_n_layers
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
        self,
        sft_model: LlamaModel,
        rlhf_model: LlamaModel,
        extrapolation_factor: float,
    ):
        if self.on_embedding:
            expo_(sft_model.embed_tokens, rlhf_model.embed_tokens, extrapolation_factor)

        if self.fix_first_n_layers == "half":
            self.fix_first_n_layers = len(sft_model.layers) // 2
        if self.fix_last_n_layers == "half":
            self.fix_last_n_layers = len(sft_model.layers) // 2

        for layer_idx in range(
            self.fix_first_n_layers, len(sft_model.layers) - self.fix_last_n_layers
        ):
            sft_layer = sft_model.layers[layer_idx]
            expo_linear_modules_(
                sft_layer.self_attn,
                rlhf_model.layers[layer_idx].self_attn,
                extrapolation_factor=extrapolation_factor
                * self.attention_scaling_factor,
                merge_dtype=torch.float32,
            )
            expo_linear_modules_(
                sft_layer.mlp,
                rlhf_model.layers[layer_idx].mlp,
                extrapolation_factor=extrapolation_factor,
                merge_dtype=torch.float32,
            )
