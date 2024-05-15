from abc import ABC, abstractmethod

from omegaconf import DictConfig


class BaseTask(ABC):
    _taskpool = None

    def __init__(self, task_config: DictConfig):
        self.config = task_config

    @abstractmethod
    def evaluate(self, model):
        """
        Evaluate the model on the task.
        Returns a dictionary containing the evaluation metrics.
        """
        raise NotImplementedError
