from omegaconf import DictConfig, open_dict

from fusion_bench.taskpool import TaskPool


class GPT2TextClassificationTaskPool(TaskPool):
    def __init__(self, taskpool_config: DictConfig):
        super().__init__(taskpool_config)
