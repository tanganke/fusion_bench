import logging

import torch

from fusion_bench import BaseModelFusionAlgorithm, BaseModelPool
from fusion_bench.utils.state_dict_arithmetic import state_dict_weighted_sum

log = logging.getLogger(__name__)


class LinearInterpolationAlgorithm(BaseModelFusionAlgorithm):
    _config_mapping = BaseModelFusionAlgorithm._config_mapping | {
        "t": "t",
    }

    def __init__(self, t: float, **kwargs):
        self.t = t
        super().__init__(**kwargs)

    def run(self, modelpool: BaseModelPool):
        assert (
            modelpool.all_model_names == 2
        ), "linear interpolation expect exactly 2 models"
        primary_model = modelpool.load_model(modelpool.all_model_names[0])
        secondary_model = modelpool.load_model(modelpool.all_model_names[1])

        with torch.no_grad():
            primary_state_dict = primary_model.state_dict()
            secondary_state_dict = secondary_model.state_dict()
            state_dict = state_dict_weighted_sum(
                [primary_state_dict, secondary_state_dict], [1 - self.t, self.t]
            )
        primary_model.load_state_dict(state_dict)
        return primary_model
