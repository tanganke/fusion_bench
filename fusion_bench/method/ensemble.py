import logging
from copy import deepcopy
from typing import List, Mapping, Union

import numpy as np
import torch
from torch import Tensor, nn

from fusion_bench.models.wrappers.ensemble import (
    EnsembleModule,
    MaxModelPredictor,
    WeightedEnsembleModule,
)

from ..modelpool import ModelPool, to_modelpool
from .base_algorithm import ModelFusionAlgorithm

log = logging.getLogger(__name__)


class EnsembleAlgorithm(ModelFusionAlgorithm):
    @torch.no_grad()
    def run(self, modelpool: ModelPool | List[nn.Module]):
        modelpool = to_modelpool(modelpool)
        log.info(f"Running ensemble algorithm with {len(modelpool)} models")

        models = [modelpool.load_model(m) for m in modelpool.model_names]
        ensemble = EnsembleModule(models=models)
        return ensemble


class WeightedEnsembleAlgorithm(ModelFusionAlgorithm):
    @torch.no_grad()
    def run(self, modelpool: ModelPool | List[nn.Module]):
        modelpool = to_modelpool(modelpool)
        log.info(f"Running weighted ensemble algorithm with {len(modelpool)} models")

        models = [modelpool.load_model(m) for m in modelpool.model_names]
        if self.config.weights is None:
            weights = np.ones(len(models)) / len(models)
        else:
            weights = self.config.weights
        ensemble = WeightedEnsembleModule(
            models,
            weights=weights,
            normalize=self.config.get("normalize", True),
        )
        return ensemble


class MaxModelPredictorAlgorithm(ModelFusionAlgorithm):
    @torch.no_grad()
    def run(self, modelpool: ModelPool | List[nn.Module]):
        modelpool = to_modelpool(modelpool)
        log.info(f"Running max predictor algorithm with {len(modelpool)} models")

        models = [modelpool.load_model(m) for m in modelpool.model_names]
        ensemble = MaxModelPredictor(models=models)
        return ensemble
