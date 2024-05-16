import logging
from copy import deepcopy
from typing import List, Mapping, Union

import torch
from torch import Tensor, nn

from ..utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
)
from ..utils.type import _StateDict
from .base_algorithm import ModelFusionAlgorithm
from ..modelpool import ModelPool

log = logging.getLogger(__name__)


class WeightedAverageAlgorithm(ModelFusionAlgorithm):

    @torch.no_grad()
    def fuse(self, modelpool: ModelPool):
        """
        Fuses the models in the model pool using a weighted average approach.

        Parameters
        ----------
        modelpool : ModelPool
            The pool of models to be fused.

        Raises
        ------
        ValueError
            If the number of weights does not match the number of models in the model pool.

        Returns
        -------
        forward_model : torch.nn.Module
            The resulting model after fusion.
        """
        log.info("Fusing models using weighted average.")
        weights = self.config.weights
        if len(weights) != len(modelpool.model_names):
            raise ValueError(
                "Number of weights must match the number of models.,"
                f"but got {len(weights)} weights and {len(modelpool.model_names)} models."
                f"weights: {weights}, models: {modelpool.model_names}"
            )
        sd: _StateDict = None
        forward_model = None

        for model_name, weight in zip(modelpool.model_names, weights):
            model = modelpool.load_model(model_name)
            if sd is None:
                sd = state_dict_mul(model.state_dict(keep_vars=True), weight)
                forward_model = model
            else:
                sd = state_dict_add(
                    sd, state_dict_mul(model.state_dict(keep_vars=True), weight)
                )

        forward_model.load_state_dict(sd)
        return forward_model
