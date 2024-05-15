from omegaconf import DictConfig

from abc import ABC, abstractmethod


class TaskPool(ABC):

    def __init__(self, taskpool_config: DictConfig):
        super().__init__()
        self.config = taskpool_config

    @abstractmethod
    def evaluate(self, model):
        raise NotImplementedError
