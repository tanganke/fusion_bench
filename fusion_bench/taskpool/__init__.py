from omegaconf import DictConfig

from .dummy import DummyTaskPool
from .base_pool import TaskPool
from .clip_image_classification import CLIPImageClassificationTaskPool


def load_taskpool(taskpool_config: DictConfig):
    if hasattr(taskpool_config, "type"):
        if taskpool_config.type == "dummy":
            return DummyTaskPool(taskpool_config)
        if taskpool_config.type == "clip_vit_classification":
            return CLIPImageClassificationTaskPool(taskpool_config)
        else:
            raise ValueError(f"Unknown task pool type: {taskpool_config.type}")
    else:
        raise ValueError("Task pool type not specified")
