"""
Base algorithm classes for model fusion.

This module provides the foundational abstract base class for implementing model fusion
algorithms in the FusionBench framework. It defines the standard interface and lifecycle
hooks that all fusion algorithms should follow.

The main class `BaseAlgorithm` serves as a template for creating various model fusion
strategies such as simple averaging, task arithmetic, weight interpolation, and more
advanced techniques. It integrates with the YAML configuration system and provides
hooks for setup and cleanup operations.

Classes:
    BaseAlgorithm: Abstract base class for all model fusion algorithms.
    BaseModelFusionAlgorithm: Alias for BaseAlgorithm (backward compatibility).

Example:
    Implementing a custom fusion algorithm:

    >>> from fusion_bench.method.base_algorithm import BaseAlgorithm
    >>> from fusion_bench.modelpool import BaseModelPool
    >>>
    >>> class WeightedAverageAlgorithm(BaseAlgorithm):
    ...     def __init__(self, weights=None, **kwargs):
    ...         self.register_parameter_to_config("weights", "weights", weights or [])
    ...         super().__init__(**kwargs)
    ...
    ...     def run(self, modelpool: BaseModelPool):
    ...         models = list(modelpool)
    ...         if len(self.weights) != len(models):
    ...             raise ValueError("Number of weights must match number of models")
    ...
    ...         # Implement weighted averaging logic here
    ...         return fused_model
"""

import logging
from abc import abstractmethod
from typing import Optional  # noqa: F401

from fusion_bench.mixins import BaseYAMLSerializable
from fusion_bench.modelpool import BaseModelPool

__all__ = ["BaseAlgorithm", "BaseModelFusionAlgorithm"]

log = logging.getLogger(__name__)


class BaseAlgorithm(BaseYAMLSerializable):
    """
    Base class for model fusion algorithms.

    This abstract class provides a standardized interface for implementing model fusion
    algorithms. It inherits from BaseYAMLSerializable to support configuration loading
    from YAML files.

    The class follows a template method pattern where subclasses must implement the
    core fusion logic in the `run` method, while optional lifecycle hooks allow for
    setup and cleanup operations.

    Attributes:
        _program: Optional program reference for algorithm execution context.
        _config_key (str): Configuration key used for YAML serialization, defaults to "method".

    Examples:
        Creating a simple averaging algorithm:

        >>> class SimpleAverageAlgorithm(BaseAlgorithm):
        ...     def run(self, modelpool: BaseModelPool):
        ...         # Implementation of model averaging logic
        ...         return averaged_model
        ...
        >>> algorithm = SimpleAverageAlgorithm()
        >>> merged_model = algorithm.run(modelpool)

        Loading algorithm from YAML configuration:

        >>> algorithm = BaseAlgorithm.from_yaml("config.yaml")
        >>> result = algorithm.run(modelpool)

    Note:
        Subclasses must implement the abstract `run` method to define the specific
        fusion strategy (e.g., simple averaging, task arithmetic, etc.).
    """

    _program = None
    _config_key = "method"

    def on_run_start(self):
        """
        Lifecycle hook called at the beginning of algorithm execution.

        This method is invoked before the main `run` method executes, providing
        an opportunity for subclasses to perform initialization tasks such as:

        - Setting up logging or monitoring
        - Initializing algorithm-specific state
        - Validating prerequisites
        - Preparing computational resources

        The default implementation does nothing, allowing subclasses to override
        as needed for their specific requirements.

        Examples:
            >>> class MyAlgorithm(BaseAlgorithm):
            ...     def on_run_start(self):
            ...         super().on_run_start()
            ...         print("Starting model fusion...")
            ...         self.start_time = time.time()
        """
        pass

    def on_run_end(self):
        """
        Lifecycle hook called at the end of algorithm execution.

        This method is invoked after the main `run` method completes, providing
        an opportunity for subclasses to perform cleanup and finalization tasks such as:

        - Logging execution statistics or results
        - Cleaning up temporary resources
        - Saving intermediate results or metrics
        - Releasing computational resources

        The method is called regardless of whether the `run` method succeeded or failed,
        making it suitable for cleanup operations that should always occur.

        The default implementation does nothing, allowing subclasses to override
        as needed for their specific requirements.

        Examples:
            >>> class MyAlgorithm(BaseAlgorithm):
            ...     def on_run_end(self):
            ...         super().on_run_end()
            ...         elapsed = time.time() - self.start_time
            ...         print(f"Fusion completed in {elapsed:.2f}s")
        """
        pass

    @abstractmethod
    def run(self, modelpool: BaseModelPool):
        """
        Execute the model fusion algorithm on the provided model pool.

        This is the core method that must be implemented by all subclasses to define
        their specific fusion strategy. The method takes a pool of models and produces
        a fused result according to the algorithm's logic.

        Args:
            modelpool (BaseModelPool): A collection of models to be fused. The modelpool
                provides access to individual models and their metadata, allowing the
                algorithm to iterate over models, access their parameters, and perform
                fusion operations.

        Returns:
            The type of return value depends on the specific algorithm implementation.
                Common return types include:

                - A single fused model (torch.nn.Module)
                - A dictionary of fused models for multi-task scenarios
                - Fusion results with additional metadata
                - Custom data structures specific to the algorithm

        Raises:
            NotImplementedError: If called on the base class without implementation.
            ValueError: If the modelpool is invalid or incompatible with the algorithm.
            RuntimeError: If fusion fails due to model incompatibilities or other issues.

        Examples:
            Simple averaging implementation:

            >>> def run(self, modelpool: BaseModelPool):
            ...     models = [model for model in modelpool]
            ...     averaged_params = {}
            ...     for name in models[0].state_dict():
            ...         averaged_params[name] = torch.stack([
            ...             model.state_dict()[name] for model in models
            ...         ]).mean(dim=0)
            ...     fused_model = copy.deepcopy(models[0])
            ...     fused_model.load_state_dict(averaged_params)
            ...     return fused_model

            Task arithmetic implementation:

            >>> def run(self, modelpool: BaseModelPool):
            ...     pretrained = modelpool.get_model('pretrained')
            ...     task_vectors = []
            ...     for model_name in modelpool.model_names:
            ...         if model_name != 'pretrained':
            ...             task_vector = self.compute_task_vector(
            ...                 modelpool.get_model(model_name), pretrained
            ...             )
            ...             task_vectors.append(task_vector)
            ...     return self.merge_task_vectors(pretrained, task_vectors)

        Note:
            - The modelpool iteration order may affect results for non-commutative operations
            - Ensure model compatibility (architecture, parameter shapes) before fusion
            - Consider memory constraints when loading multiple large models
            - Use appropriate device placement for GPU/CPU computation
        """
        pass


BaseModelFusionAlgorithm = BaseAlgorithm
"""
Alias for BaseAlgorithm class.

This alias is provided for backward compatibility and semantic clarity.
Some users may prefer the more explicit name 'BaseModelFusionAlgorithm'
to emphasize that this class is specifically designed for model fusion
tasks, while others may prefer the shorter 'BaseAlgorithm' name.

Both names refer to the exact same class and can be used interchangeably.

Examples:
    Using the original name:
    >>> class MyAlgorithm(BaseAlgorithm):
    ...     def run(self, modelpool): pass

    Using the alias:
    >>> class MyAlgorithm(BaseModelFusionAlgorithm):
    ...     def run(self, modelpool): pass

Note:
    The alias is maintained for compatibility but BaseAlgorithm is the
    preferred name for new implementations.
"""
