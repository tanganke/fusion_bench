from fusion_bench.method import AlgorithmFactory
from fusion_bench.modelpool import ModelPoolFactory
from fusion_bench.taskpool import TaskPoolFactory

__all__ = ["AVAILABLE_ALGORITHMS", "AVAILABLE_MODELPOOLS", "AVAILABLE_TASKPOOLS"]

AVAILABLE_ALGORITHMS = AlgorithmFactory.available_algorithms()
AVAILABLE_MODELPOOLS = ModelPoolFactory.available_modelpools()
AVAILABLE_TASKPOOLS = TaskPoolFactory.available_taskpools()
