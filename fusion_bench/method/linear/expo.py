"""
This module contains the implementation of ExPO merge for general nn.Modules.

Reference:
- Zheng et al. Weak-to-Strong Extrapolation Expedites Alignment.
"""

import logging
from copy import deepcopy

import torch
from torch import nn

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.method import SimpleAverageAlgorithm
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
)

log = logging.getLogger(__name__)


def expo_merge(
    sft_model: nn.Module,
    rlhf_model: nn.Module,
    extrapolation_factor: float,
    inplace: bool = True,
    enable_grad: bool = False,
):
    """
    Minimal implementation of ExPO merge.

    Args:
        sft_model (nn.Module): The pretrained model (base model).
        rlhf_model (nn.Module): The finetuned model (medium-aligned model).
        extrapolation_factor (float): The extrapolation factor.
        inplace (bool): Whether to perform the merge in-place. If not, the rlhf_model will be copied before merging.
        enable_grad (bool): Whether to enable gradient computation during the merge.

    Returns:
        nn.Module: The merged model.
    """

    if not inplace:
        rlhf_model = deepcopy(rlhf_model)

    with torch.set_grad_enabled(enable_grad):
        for (sft_name, sft_param), (rlhf_name, rlhf_param) in zip(
            sft_model.named_parameters(), rlhf_model.named_parameters()
        ):
            assert sft_name == rlhf_name, f"Model mismatch: {sft_name} != {rlhf_name}"
            rlhf_param.data = rlhf_param.data + extrapolation_factor * (
                rlhf_param.data - sft_param.data
            )
    return rlhf_model


class ExPOAlgorithm(BaseAlgorithm):
    R"""
    ExPO merge algorithm.

    This algorithm merges a pretrained model with a finetuned model.

    $$\theta_{merged} = \theta_{sft} + \alpha (\theta_{rlhf} - \theta_{sft})$$

    where $\theta_{merged}$ is the merged model, $\theta_{rlhf}$ is the finetuned model (medium-aligned model),
    $\theta_{sft}$ is the pretrained model (base model), and $\alpha$ is the extrapolation factor.

    In the configuration, the SFT model should have name `_pretrained_` and the rlhf name can be set arbitarily.
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "extrapolation_factor": "extrapolation_factor"
    }

    def __init__(self, extrapolation_factor: float, **kwargs):
        self.extrapolation_factor = extrapolation_factor
        super().__init__(**kwargs)

    def run(self, modelpool: BaseModelPool):
        """
        Run the ExPO merge algorithm.

        Args:
            modelpool (BaseModelPool): The pool of models to merge.

        Returns:
            nn.Module: The merged model.
        """
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        assert len(modelpool.model_names) >= 1, "ExPO requires at least one model."
        assert modelpool.has_pretrained, "ExPO requires pretrained models (base model)."

        sft_model = modelpool.load_pretrained_model()
        if len(modelpool) == 1:
            rlhf_model = modelpool.load_model(modelpool.model_names[0])
        else:
            # if there are multiple RLHF models, use simple average to merge them before running ExPO
            log.info(
                f"There are {len(modelpool)} models in the model pool, averaging them first..."
            )
            rlhf_model = SimpleAverageAlgorithm().run(modelpool)

        # merge the pretrained model and the finetuned model
        delta_parameters = state_dict_sub(
            rlhf_model.state_dict(), sft_model.state_dict()
        )
        merged_sd = state_dict_add(
            rlhf_model.state_dict(),
            state_dict_mul(delta_parameters, scalar=self.extrapolation_factor),
        )

        rlhf_model.load_state_dict(merged_sd)
        return rlhf_model
