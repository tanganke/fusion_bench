from omegaconf import DictConfig

from .base_pool import TaskPool
from .clip_image_classification import CLIPImageClassificationTaskPool
from .dummy import DummyTaskPool
from .flan_t5_glue_text_generation import FlanT5GLUETextGenerationTaskPool
from .gpt2_text_classification import GPT2TextClassificationTaskPool


def load_taskpool_from_config(taskpool_config: DictConfig):
    """
    Loads a task pool based on the provided configuration.

    The function checks the 'type' attribute of the configuration and returns an instance of the corresponding task pool.
    If the 'type' attribute is not found or does not match any known task pool types, a ValueError is raised.

    Args:
        taskpool_config (DictConfig): The configuration for the task pool. Must contain a 'type' attribute that specifies the type of the task pool.

    Returns:
        An instance of the specified task pool.

    Raises:
        ValueError: If 'type' attribute is not found in the configuration or does not match any known task pool types.
    """
    if hasattr(taskpool_config, "type"):
        if taskpool_config.type == "dummy":
            return DummyTaskPool(taskpool_config)
        elif taskpool_config.type == "clip_vit_classification":
            return CLIPImageClassificationTaskPool(taskpool_config)
        elif taskpool_config.type == "GPT2TextClassificationTaskPool":
            return GPT2TextClassificationTaskPool(taskpool_config)
        elif taskpool_config.type == "FlanT5GLUETextGenerationTaskPool":
            return FlanT5GLUETextGenerationTaskPool(taskpool_config)
        else:
            raise ValueError(f"Unknown task pool type: {taskpool_config.type}")
    else:
        raise ValueError("Task pool type not specified")
