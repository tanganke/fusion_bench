import logging

import torch
from typing_extensions import override

from fusion_bench.method import BaseModelFusionAlgorithm
from fusion_bench.modelpool import BaseModelPool

from .slerp_utils import slerp

log = logging.getLogger(__name__)


def slerp_on_state_dicts(
    t,
    primary_state_dict,
    secondary_state_dict,
    *,
    DOT_THRESHOLD: float = 0.9995,
    epsilon: float = 1e-8,
):
    state_dict = {}
    for key in secondary_state_dict:
        v0 = primary_state_dict[key]
        v1 = secondary_state_dict[key]
        if v0.shape != v1.shape:
            log.warning(
                f"Skipping key {key} because the shapes of the tensors are different: {v0.shape} vs {v1.shape}. Base model parameters will be used."
            )
            state_dict[key] = v0
        else:
            state_dict[key] = slerp(t, v0, v1, DOT_THRESHOLD, epsilon)
    return state_dict


class SlerpMergeAlgorithm(BaseModelFusionAlgorithm):
    """
    General purpose implementation of Slerp (Spherical Linear Interpolation) for PyTorch models.
    """

    _config_mapping = BaseModelFusionAlgorithm._config_mapping + {
        "t": "t",
        "DOT_THRESHOLD": "DOT_THRESHOLD",
        "epsilon": "epsilon",
    }

    def __init__(self, t: float, DOT_THRESHOLD: float = 0.9995, epsilon: float = 1e-8):
        """
        Args:
            t (float): The interpolation parameter. Must be in the range [0, 1].
            DOT_THRESHOLD (float, optional): The threshold for the dot product of the two vectors. Defaults to 0.9995.
            epsilon (float, optional): The epsilon value for numerical stability. Defaults to 1e-8.
        """
        self.t = t
        self.DOT_THRESHOLD = DOT_THRESHOLD
        self.epsilon = epsilon
        super().__init__()

    @override
    def run(self, modelpool: BaseModelPool):
        assert len(modelpool.all_model_names) == 2, "Slerp expect exactly 2 models"
        primary_model = modelpool.load_model(modelpool.all_model_names[0])
        secondary_model = modelpool.load_model(modelpool.all_model_names[1])

        with torch.no_grad():
            primary_state_dict = primary_model.state_dict()
            secondary_state_dict = secondary_model.state_dict()
            state_dict = slerp_on_state_dicts(
                self.t,
                primary_state_dict,
                secondary_state_dict,
                DOT_THRESHOLD=self.DOT_THRESHOLD,
                epsilon=self.epsilon,
            )

        primary_model.load_state_dict(state_dict)
        return primary_model
