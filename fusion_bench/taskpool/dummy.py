"""
Dummy task pool implementation for debugging and testing purposes.

This module provides a minimal task pool implementation that can be used for
debugging model fusion workflows, testing infrastructure, and validating model
architectures without running expensive evaluation procedures. It's particularly
useful during development and prototyping phases.
"""

from typing import Optional

from lightning.pytorch.utilities import rank_zero_only
from torch import nn

from fusion_bench.models.separate_io import separate_save
from fusion_bench.taskpool.base_pool import BaseTaskPool
from fusion_bench.utils import timeit_context
from fusion_bench.utils.parameters import count_parameters, print_parameters


def get_model_summary(model: nn.Module) -> dict:
    """Generate a comprehensive summary report for a PyTorch model.

    Analyzes the given model to extract key information about its architecture,
    parameter count, and training characteristics. This function is useful for
    model introspection and comparative analysis during model fusion workflows.

    The summary includes both trainable and total parameter counts, which helps
    in understanding model complexity and memory requirements. The trainable
    percentage is particularly useful for identifying models with frozen layers
    or parameter-efficient fine-tuning setups.

    Args:
        model: The PyTorch model to analyze. Can be any nn.Module instance
            including complex models, fusion models, or pre-trained models.

    Returns:
        dict: A structured report containing model information:
            - model_info: Dictionary with parameter statistics
                - trainable_params: Number of trainable parameters
                - all_params: Total number of parameters (trainable + frozen)
                - trainable_percentage: Ratio of trainable to total parameters

    Example:
        ```python
        >>> model = MyModel()
        >>> summary = get_model_summary(model)
        >>> print(summary)
        {
            "model_info": {
                "trainable_params": 1234567,
                "all_params": 1234567,
                "trainable_percentage": 1.0
            }
        }
        ```
    """
    report = {}
    training_params, all_params = count_parameters(model)
    report["model_info"] = {
        "trainable_params": training_params,
        "all_params": all_params,
        "trainable_percentage": training_params / all_params,
    }
    return report


class DummyTaskPool(BaseTaskPool):
    """A lightweight task pool implementation for debugging and development workflows.

    This dummy task pool provides a minimal evaluation interface that focuses on
    model introspection rather than task-specific performance evaluation. It's
    designed for development scenarios where you need to test model fusion
    pipelines, validate architectures, or debug workflows without the overhead
    of running actual evaluation tasks.

    The task pool is particularly useful when:
        - You want to verify model fusion works correctly
        - You need to check parameter counts after fusion
        - You're developing new fusion algorithms
        - You want to test infrastructure without expensive evaluations

    Example:
        ```python
        >>> taskpool = DummyTaskPool(model_save_path="/tmp/fused_model")
        >>> results = taskpool.evaluate(fused_model)
        >>> print(f"Model has {results['model_info']['trainable_params']} parameters")
        ```
    """

    def __init__(self, model_save_path: Optional[str] = None, **kwargs):
        """Initialize the dummy task pool with optional model saving capability.

        Args:
            model_save_path: Optional path where the evaluated model should be saved.
                If provided, the model will be serialized and saved to this location
                after evaluation using the separate_save utility. If None, no model
                saving will be performed.

        Example:
            ```python
            >>> # Create taskpool without saving
            >>> taskpool = DummyTaskPool()

            >>> # Create taskpool with model saving
            >>> taskpool = DummyTaskPool(model_save_path="/path/to/save/model.pth")
            ```
        """
        super().__init__(**kwargs)
        self.model_save_path = model_save_path

    def evaluate(self, model):
        """Perform lightweight evaluation and analysis of the given model.

        This method provides a minimal evaluation that focuses on model introspection
        rather than task-specific performance metrics. It performs parameter analysis,
        optionally saves the model, and returns a summary report.

        The evaluation process includes:
        1. Printing human-readable parameter information (rank-zero only)
        2. Optionally saving the model if a save path was configured
        3. Generating and returning a model summary report

        Args:
            model: The model to evaluate. Can be any PyTorch nn.Module including
                fusion models, pre-trained models, or custom architectures.

        Returns:
            dict: A model summary report containing parameter statistics and
                architecture information. See get_model_summary() for detailed
                format specification.

        Example:
            ```python
            >>> taskpool = DummyTaskPool(model_save_path="/tmp/model.pth")
            >>> model = torch.nn.Linear(10, 5)
            >>> results = taskpool.evaluate(model)
            >>> print(f"Trainable params: {results['model_info']['trainable_params']}")
            ```
        """
        if rank_zero_only.rank == 0:
            print_parameters(model, is_human_readable=True)

            if self.model_save_path is not None:
                with timeit_context(f"Saving the model to {self.model_save_path}"):
                    separate_save(model, self.model_save_path)

        return get_model_summary(model)
