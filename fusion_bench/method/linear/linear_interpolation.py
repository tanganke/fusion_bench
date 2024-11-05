import logging

import torch

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.utils.state_dict_arithmetic import state_dict_weighted_sum

log = logging.getLogger(__name__)


class LinearInterpolationAlgorithm(BaseAlgorithm):
    R"""
    LinearInterpolationAlgorithm performs linear interpolation between two models.
    Returns a model with the state dict that is a linear interpolation of the state dicts of the two models.
    $\theta = (1-t) \theta_1 + t \theta_2$
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "t": "t",
    }

    def __init__(self, t: float, **kwargs):
        """
        Initialize the LinearInterpolationAlgorithm with the given interpolation parameter.

        Args:
            t (float): The interpolation parameter, should be in the range [0, 1].
            **kwargs: Additional keyword arguments.
        """
        assert 0 <= t <= 1, "t should be in the range [0, 1]"
        self.t = t
        super().__init__(**kwargs)

    def run(self, modelpool: BaseModelPool):
        """
        Run the linear interpolation algorithm on the given model pool.

        This method performs linear interpolation between two models in the model pool
        and returns a model with the interpolated state dict.

        Args:
            modelpool (BaseModelPool): The pool of models to interpolate. Must contain exactly two models.

        Returns:
            nn.Module: The model with the interpolated state dict.
        """
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
