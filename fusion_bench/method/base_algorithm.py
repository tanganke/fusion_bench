import logging
from abc import ABC, abstractmethod
from typing import Optional

from omegaconf import DictConfig

from fusion_bench.mixins import BaseYAMLSerializableModel
from fusion_bench.modelpool import BaseModelPool

__all__ = ["BaseModelFusionAlgorithm"]

log = logging.getLogger(__name__)


class BaseModelFusionAlgorithm(BaseYAMLSerializableModel):
    _program = None

    @abstractmethod
    def run(self, modelpool: BaseModelPool):
        """
        Fuse the models in the given model pool.

        Examples:
            >>> algorithm = SimpleAverageAlgorithm()
            >>> modelpool = ModelPool()
            >>> merged_model = algorithm.fuse(modelpool)

        Args:
            modelpool (_type_): _description_
        """
        pass
