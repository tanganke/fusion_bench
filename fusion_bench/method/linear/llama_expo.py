import logging
from typing import cast

from torch import nn
from transformers import LlamaForCausalLM

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.method import SimpleAverageAlgorithm
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
)

log = logging.getLogger(__name__)


def expo_(sft_model: nn.Module, rlhf_model: nn.Module, extrapolation_factor: float):
    delta_parameters = state_dict_sub(rlhf_model.state_dict(), sft_model.state_dict())
    merged_sd = state_dict_add(
        rlhf_model.state_dict(),
        state_dict_mul(delta_parameters, scalar=extrapolation_factor),
    )

    rlhf_model.load_state_dict(merged_sd)
    return rlhf_model


class ExPOAlgorithmForLlama(BaseAlgorithm):

    def __init__(
        self,
        extrapolation_factor: float,
        only_on_backbone: bool = True,
        on_linear_weights: bool = True,
        on_linear_bias: bool = False,
        **kwargs,
    ):
        self.extrapolation_factor = extrapolation_factor
        self.only_on_backbone = only_on_backbone
        self.on_linear_weights = on_linear_weights
        self.on_linear_bias = on_linear_bias
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

        if self.only_on_backbone:
            rlhf_model.model = expo_(
                sft_model.model, rlhf_model.model, self.extrapolation_factor
            )
        else:
            rlhf_model = expo_(sft_model, rlhf_model, self.extrapolation_factor)
        return rlhf_model
