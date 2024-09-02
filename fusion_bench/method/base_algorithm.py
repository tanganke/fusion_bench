from abc import ABC, abstractmethod
from typing import Optional

from omegaconf import DictConfig

from fusion_bench.mixins import YAMLSerializationMixin
from fusion_bench.modelpool import BaseModelPool

__all__ = ["ModelFusionAlgorithm"]


class ModelFusionAlgorithm(ABC):
    def __init__(self, algorithm_config: Optional[DictConfig] = None):
        super().__init__()
        if algorithm_config is None:
            algorithm_config = DictConfig({})
        self.config = algorithm_config

    @abstractmethod
    def run(self, modelpool):
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
