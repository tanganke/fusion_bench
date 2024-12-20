from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from omegaconf import DictConfig

if TYPE_CHECKING:
    from fusion_bench.programs.base_program import BaseHydraProgram

__all__ = ["ModelFusionAlgorithm"]


class ModelFusionAlgorithm(ABC):
    """
    Abstract base class for model fusion algorithms (for v0.1.x versions, deprecated).
    For implementing new method, use `fusion_bench.method.BaseModelFusionAlgorithm` instead.

    This class provides a template for implementing model fusion algorithms.
    Subclasses must implement the `run` method to define the fusion logic.

    Attributes:
        config (DictConfig): Configuration for the algorithm.
    """

    _program: "BaseHydraProgram" = None
    """A reference to the program that is running the algorithm."""

    def __init__(self, algorithm_config: Optional[DictConfig] = None):
        """
        Initialize the model fusion algorithm with the given configuration.

        Args:
            algorithm_config (Optional[DictConfig]): Configuration for the algorithm. Defaults to an empty configuration if not provided.
                Get access to the configuration using `self.config`.
        """
        if algorithm_config is None:
            algorithm_config = DictConfig({})
        self.config = algorithm_config

    @abstractmethod
    def run(self, modelpool):
        """
        Fuse the models in the given model pool.

        This method must be implemented by subclasses to define the fusion logic.
        `modelpool` is an object responsible for managing the models to be fused and optional datasets to be used for fusion.

        Args:
            modelpool: The pool of models to fuse.

        Returns:
            The fused model.

        Examples:
            >>> algorithm = SimpleAverageAlgorithm()
            >>> modelpool = ModelPool()
            >>> merged_model = algorithm.fuse(modelpool)
        """
        pass
