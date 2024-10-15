import logging
from copy import deepcopy
from typing import List, Mapping, Union

import numpy as np
import torch
from torch import Tensor, nn

from fusion_bench.method import BaseModelFusionAlgorithm
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.models.wrappers.ensemble import (
    EnsembleModule,
    MaxModelPredictor,
    WeightedEnsembleModule,
)

log = logging.getLogger(__name__)


class EnsembleAlgorithm(BaseModelFusionAlgorithm):
    @torch.no_grad()
    def run(self, modelpool: BaseModelPool | List[nn.Module]):
        log.info(f"Running ensemble algorithm with {len(modelpool)} models")

        models = [modelpool.load_model(m) for m in modelpool.model_names]
        ensemble = EnsembleModule(models=models)
        return ensemble


class WeightedEnsembleAlgorithm(BaseModelFusionAlgorithm):
    @torch.no_grad()
    def run(self, modelpool: BaseModelPool | List[nn.Module]):
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(models=modelpool)

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


class MaxModelPredictorAlgorithm(BaseModelFusionAlgorithm):
    @torch.no_grad()
    def run(self, modelpool: BaseModelPool | List[nn.Module]):
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(models=modelpool)

        log.info(f"Running max predictor algorithm with {len(modelpool)} models")

        models = [modelpool.load_model(m) for m in modelpool.model_names]
        ensemble = MaxModelPredictor(models=models)
        return ensemble
