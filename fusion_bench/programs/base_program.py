from abc import abstractmethod

from fusion_bench.mixins import BaseYAMLSerializableModel


class BaseHydraProgram(BaseYAMLSerializableModel):
    @abstractmethod
    def run(self):
        pass
