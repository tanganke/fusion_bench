"""
This is the dummy task pool that is used for debugging purposes.
"""

from omegaconf import DictConfig

from fusion_bench.utils.parameters import print_parameters

from .base_pool import TaskPool


class DummyTaskPool(TaskPool):
    """
    This is a dummy task pool used for debugging purposes. It inherits from the base TaskPool class.
    """

    def evaluate(self, model):
        """
        Evaluate the given model.
        This method does nothing but print the parameters of the model in a human-readable format.

        Args:
            model: The model to evaluate.
        """
        print_parameters(model, is_human_readable=True)
