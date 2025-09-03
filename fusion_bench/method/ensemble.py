import logging
from typing import List, Mapping, Optional, Union  # noqa: F401

import numpy as np
import torch
from torch import nn

from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import auto_register_config
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.models.wrappers.ensemble import (
    EnsembleModule,
    MaxModelPredictor,
    WeightedEnsembleModule,
)

log = logging.getLogger(__name__)


@auto_register_config
class SimpleEnsembleAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        device_map: Optional[Mapping[int, Union[str, torch.device]]] = None,
        **kwargs,
    ):
        """
        Initializes the SimpleEnsembleAlgorithm with an optional device map.

        Args:
            device_map (Optional[Mapping[int, Union[str, torch.device]]], optional): A mapping from model index to device. Defaults to None.
        """
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool | List[nn.Module]) -> EnsembleModule:
        """
        Run the simple ensemble algorithm on the given model pool.

        Args:
            modelpool (BaseModelPool | List[nn.Module]): The pool of models to ensemble.

        Returns:
            EnsembleModule: The ensembled model.
        """
        log.info(f"Running ensemble algorithm with {len(modelpool)} models")
        models = [modelpool.load_model(m) for m in modelpool.model_names]

        log.info("creating ensemble module")
        ensemble = EnsembleModule(models=models, device_map=self.device_map)
        return ensemble


@auto_register_config
class WeightedEnsembleAlgorithm(BaseAlgorithm):

    def __init__(
        self,
        normalize: bool = True,
        weights: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool | List[nn.Module]) -> WeightedEnsembleModule:
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
    def run(self, modelpool: BaseModelPool | List[nn.Module]) -> MaxModelPredictor:
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
