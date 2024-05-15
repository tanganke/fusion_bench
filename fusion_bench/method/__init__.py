from omegaconf import DictConfig
from .simple_average import SimpleAverageAlgorithm


def load_algorithm(method_config: DictConfig):
    if method_config.name == "simple_average":
        return SimpleAverageAlgorithm(method_config)
    else:
        raise ValueError(f"Unknown algorithm: {method_config.name}")
