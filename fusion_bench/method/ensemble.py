import logging
from copy import deepcopy
from typing import List, Mapping, Union

import torch
from torch import Tensor, nn

from fusion_bench.models.wrappers.ensemble import (
    EnsembleModule,
    MaxPredictor,
    WeightedEnsembleModule,
)

from ..modelpool import ModelPool
from ..utils.state_dict_arithmetic import state_dict_add, state_dict_mul
from ..utils.type import _StateDict
from .base_algorithm import ModelFusionAlgorithm
import numpy as np

log = logging.getLogger(__name__)


class EnsembleAlgorithm(ModelFusionAlgorithm):

    @torch.no_grad()
    def run(self, modelpool: ModelPool):
        log.info(f"Running ensemble algorithm with {len(modelpool)} models")

        models = [modelpool.load_model(m) for m in modelpool.model_names]
        ensemble = EnsembleModule(models=models)
        return ensemble


class WeightedEnsembleAlgorithm(ModelFusionAlgorithm):

    @torch.no_grad()
    def run(self, modelpool: ModelPool):
        log.info(f"Running weighted ensemble algorithm with {len(modelpool)} models")

        models = [modelpool.load_model(m) for m in modelpool.model_names]
        if self.config.weights is None:
            weights = np.ones(len(models)) / len(models)
        else:
            weights = self.config.weights
        ensemble = WeightedEnsembleModule(models, weights=weights)
        return ensemble


class MaxPredictorAlgorithm(ModelFusionAlgorithm):

    @torch.no_grad()
    def run(self, modelpool: ModelPool):
        log.info(f"Running max predictor algorithm with {len(modelpool)} models")

        models = [modelpool.load_model(m) for m in modelpool.model_names]
        ensemble = MaxPredictor(models=models)
        return ensemble
