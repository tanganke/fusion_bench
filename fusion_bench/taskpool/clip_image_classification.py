from omegaconf import DictConfig

from .base_pool import TaskPool


class CLIPImageClassificationTaskPool(TaskPool):
    def __init__(self, taskpool_config: DictConfig):
        super().__init__(taskpool_config)

    def evaluate(self, model):
        """
        Evaluate the model on the image classification task.
        """
        raise NotImplementedError
