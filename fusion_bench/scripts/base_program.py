from abc import abstractmethod

from fusion_bench.mixins import YAMLSerializationMixin


class BaseHydraProgram(YAMLSerializationMixin):
    @abstractmethod
    def run(self):
        pass
