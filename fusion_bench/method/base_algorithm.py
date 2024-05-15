from abc import ABC, abstractmethod


class ModelFusionAlgorithm(ABC):
    def __init__(self, algorithm_config):
        super().__init__()
        self.config = algorithm_config

    @abstractmethod
    def fuse(self, modelpool):
        pass
