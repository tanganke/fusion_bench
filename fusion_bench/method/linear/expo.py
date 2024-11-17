"""
This module contains the implementation of ExPO merge for general nn.Modules.

Reference:
- Zheng et al. Weak-to-Strong Extrapolation Expedites Alignment.
"""

import logging

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.method import SimpleAverageAlgorithm
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
)

log = logging.getLogger(__name__)


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
