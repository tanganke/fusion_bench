from omegaconf import DictConfig

from .dummy import DummyAlgorithm
from .simple_average import SimpleAverageAlgorithm
from .weighted_average import WeightedAverageAlgorithm
from .task_arithmetic import TaskArithmeticAlgorithm


def load_algorithm_from_config(method_config: DictConfig):
    if method_config.name == "dummy":
        return DummyAlgorithm(method_config)
    elif method_config.name == "simple_average":
        return SimpleAverageAlgorithm(method_config)
    elif method_config.name == "weighted_average":
        return WeightedAverageAlgorithm(method_config)
    elif method_config.name == "task_arithmetic":
        return TaskArithmeticAlgorithm(method_config)
    else:
        raise ValueError(f"Unknown algorithm: {method_config.name}")
