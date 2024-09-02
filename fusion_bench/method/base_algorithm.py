from abc import ABC, abstractmethod
from typing import Optional

from omegaconf import DictConfig

from fusion_bench.mixins import YAMLSerializationMixin
from fusion_bench.modelpool import BaseModelPool

__all__ = ["BaseModelFusionAlgorithm"]


class BaseModelFusionAlgorithm(YAMLSerializationMixin):
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
