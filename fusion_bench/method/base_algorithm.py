from abc import ABC, abstractmethod


class ModelFusionAlgorithm(ABC):
    @abstractmethod
    def fuse(self, modelpool):
        pass
