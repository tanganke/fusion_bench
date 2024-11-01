"""
This is the dummy task pool that is used for debugging purposes.
"""

from typing import Optional

from torch import nn

from fusion_bench.models.separate_io import separate_save
from fusion_bench.taskpool.base_pool import BaseTaskPool
from fusion_bench.utils import timeit_context
from fusion_bench.utils.parameters import count_parameters, print_parameters


def get_model_summary(model: nn.Module) -> dict:
    """
    Generate a report for the given model.

    Args:
        model: The model to generate the report for.

    Returns:
        dict: The generated report.
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
    """
    This is a dummy task pool used for debugging purposes. It inherits from the base TaskPool class.
    """

    def __init__(self, model_save_path: Optional[str] = None):
        super().__init__()
        self.model_save_path = model_save_path

    def evaluate(self, model):
        """
        Evaluate the given model.
        This method does nothing but print the parameters of the model in a human-readable format.

        Args:
            model: The model to evaluate.
        """
        print_parameters(model, is_human_readable=True)

        if self.model_save_path is not None:
            with timeit_context(f"Saving the model to {self.model_save_path}"):
                separate_save(model, self.model_save_path)

        return get_model_summary(model)
