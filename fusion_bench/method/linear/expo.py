"""
This module contains the implementation of ExPO merge.

Reference:
- Zheng et al. Weak-to-Strong Extrapolation Expedites Alignment.
"""

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
)


class ExPOAlgorithm(BaseAlgorithm):
    R"""
    ExPO merge algorithm.

    This algorithm merges a pretrained model with a finetuned model.

    $$\theta_{merged} = \theta_{ft} + \alpha (\theta_{ft} - \theta_{pre})$$

    where $\theta_{merged}$ is the merged model, $\theta_{ft}$ is the finetuned model (medium-aligned model),
    $\theta_{pre}$ is the pretrained model (base model), and $\alpha$ is the extrapolation factor.
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

        assert len(modelpool.all_model_names) == 2, "ExPO requires exactly two models."
        assert modelpool.has_pretrained, "ExPO requires pretrained models (base model)."

        pretrained_model = modelpool.load_pretrained_model()
        finetuned_model = modelpool.load_model(modelpool.model_names[0])

        # merge the pretrained model and the finetuned model
        delta_sd = state_dict_sub(
            finetuned_model.state_dict(), pretrained_model.state_dict()
        )
        merged_sd = state_dict_add(
            finetuned_model.state_dict(),
            state_dict_mul(delta_sd, scalar=self.extrapolation_factor),
        )

        pretrained_model.load_state_dict(merged_sd)
        return pretrained_model
