from omegaconf import DictConfig

from fusion_bench.taskpool.dummy import DummyTaskPool

from .base_pool import TaskPool


class TaskPoolFactory:
    _taskpool_types = {
        "dummy": DummyTaskPool,
        "clip_vit_classification": ".clip_image_classification.CLIPImageClassificationTaskPool",
        "FlanT5GLUETextGenerationTaskPool": ".flan_t5_glue_text_generation.FlanT5GLUETextGenerationTaskPool",
        "NYUv2TaskPool": ".nyuv2_taskpool.NYUv2TaskPool",
    }

    @staticmethod
    def create_taskpool(taskpool_config: DictConfig):
        from fusion_bench.utils import import_object

        taskpool_type = taskpool_config.get("type")
        if taskpool_type is None:
            raise ValueError("Task pool type not specified")

        if taskpool_type not in TaskPoolFactory._taskpool_types:
            raise ValueError(
                f"Unknown task pool: {taskpool_type}, available task pools: {TaskPoolFactory._taskpool_types.keys()}. You can register a new task pool using `TaskPoolFactory.register_taskpool()` method."
            )
        taskpool_cls = TaskPoolFactory._taskpool_types[taskpool_type]
        if isinstance(taskpool_cls, str):
            if taskpool_cls.startswith("."):
                taskpool_cls = f"fusion_bench.compat.taskpool.{taskpool_cls[1:]}"
            taskpool_cls = import_object(taskpool_cls)
        return taskpool_cls(taskpool_config)

    @staticmethod
    def register_taskpool(name: str, taskpool_cls):
        TaskPoolFactory._taskpool_types[name] = taskpool_cls

    @classmethod
    def available_taskpools(cls):
        return list(cls._taskpool_types.keys())


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
    return TaskPoolFactory.create_taskpool(taskpool_config)
