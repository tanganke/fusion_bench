import logging
from abc import abstractmethod
from typing import Optional  # noqa: F401

from fusion_bench.mixins import BaseYAMLSerializableModel
from fusion_bench.modelpool import BaseModelPool

__all__ = ["BaseAlgorithm", "BaseModelFusionAlgorithm"]

log = logging.getLogger(__name__)


class BaseAlgorithm(BaseYAMLSerializableModel):
    """
    Base class for model fusion algorithms.

    This class provides a template for implementing model fusion algorithms.
    Subclasses must implement the `run` method to define the fusion logic.
    """

    _program = None

    @abstractmethod
    def run(self, modelpool: BaseModelPool):
        """
        Fuse the models in the given model pool.

        This method must be implemented by subclasses to define the fusion logic.

        Examples:
            >>> algorithm = SimpleAverageAlgorithm()
            >>> modelpool = ModelPool()
            >>> merged_model = algorithm.run(modelpool)

        Args:
            modelpool (BaseModelPool): The pool of models to fuse.
        """
        pass


BaseModelFusionAlgorithm = BaseAlgorithm
"""
Alias for `BaseAlgorithm`.
"""
