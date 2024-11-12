import logging
from typing import List, Mapping, Union  # noqa: F401

import numpy as np
import torch
from torch import nn

from fusion_bench.method import BaseAlgorithm
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.models.wrappers.ensemble import (
    EnsembleModule,
    MaxModelPredictor,
    WeightedEnsembleModule,
)

log = logging.getLogger(__name__)


class SimpleEnsembleAlgorithm(BaseAlgorithm):
    @torch.no_grad()
    def run(self, modelpool: BaseModelPool | List[nn.Module]):
        """
        Run the simple ensemble algorithm on the given model pool.

        Args:
            modelpool (BaseModelPool | List[nn.Module]): The pool of models to ensemble.

        Returns:
            EnsembleModule: The ensembled model.
        """
        log.info(f"Running ensemble algorithm with {len(modelpool)} models")

        models = [modelpool.load_model(m) for m in modelpool.model_names]
        ensemble = EnsembleModule(models=models)
        return ensemble


class WeightedEnsembleAlgorithm(BaseAlgorithm):

    _config_mapping = BaseAlgorithm._config_mapping | {
        "normalize": "normalize",
        "weights": "weights",
    }

    def __init__(self, normalize: bool, weights: List[float], **kwargs):
        self.normalize = normalize
        self.weights = weights
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool | List[nn.Module]):
        """
        Run the weighted ensemble algorithm on the given model pool.

        Args:
            modelpool (BaseModelPool | List[nn.Module]): The pool of models to ensemble.

        Returns:
            WeightedEnsembleModule: The weighted ensembled model.
        """
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(models=modelpool)

        log.info(f"Running weighted ensemble algorithm with {len(modelpool)} models")

        models = [modelpool.load_model(m) for m in modelpool.model_names]
        if self.weights is None:
            weights = np.ones(len(models)) / len(models)
        else:
            weights = self.weights
        ensemble = WeightedEnsembleModule(
            models,
            weights=weights,
            normalize=self.config.get("normalize", True),
        )
        return ensemble


class MaxModelPredictorAlgorithm(BaseAlgorithm):
    @torch.no_grad()
    def run(self, modelpool: BaseModelPool | List[nn.Module]):
        """
        Run the max model predictor algorithm on the given model pool.

        Args:
            modelpool (BaseModelPool | List[nn.Module]): The pool of models to ensemble.

        Returns:
            MaxModelPredictor: The max model predictor ensembled model.
        """
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(models=modelpool)

        log.info(f"Running max predictor algorithm with {len(modelpool)} models")

        models = [modelpool.load_model(m) for m in modelpool.model_names]
        ensemble = MaxModelPredictor(models=models)
        return ensemble
