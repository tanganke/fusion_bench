import functools
import itertools
import logging
from copy import deepcopy

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    default_data_collator,
)

from fusion_bench.tasks import BaseTask
from fusion_bench.tasks.flan_t5_text_generation.glue_evaluation import (
    evaluate_accuracy,
    evaluate_spearman_rho,
)
from fusion_bench.tasks.flan_t5_text_generation.glue_load_dataset import (
    load_glue_dataset,
)
from fusion_bench.utils.parameters import count_parameters

from .base_pool import TaskPool

log = logging.getLogger(__name__)

CLASSIFICATION_TASKS = [
    "cola",
    "mnli",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
]
REGRESSION_TASKS = ["stsb"]


class FlanT5GLUETextGenerationTask(BaseTask):
    _taskpool: "FlanT5GLUETextGenerationTaskPool" = None

    @property
    def taskpool(self):
        if self._taskpool is not None:
            return self._taskpool
        else:
            raise ValueError("Taskpool not set")

    @property
    def fabric(self):
        return self.taskpool.fabric

    @property
    def tokenizer(self):
        return self.taskpool.tokenizer

    @functools.cached_property
    def dataset(self):
        log.info(f'Loading dataset: "{self.config.dataset.name}"')
        dataset = load_glue_dataset(
            self.config.dataset.name, self.tokenizer, self.taskpool.config.cache_dir
        )
        return dataset

    @functools.cached_property
    def test_dataset(self):
        return self.dataset[self.config.dataset.split]

    @property
    def test_loader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.taskpool.config.batch_size,
            num_workers=self.taskpool.config.num_workers,
            shuffle=False,
            collate_fn=default_data_collator,
        )
        loader = self.fabric.setup_dataloaders(loader)
        return loader


class FlanT5GLUETextGenerationClassificationTask(FlanT5GLUETextGenerationTask):
    @torch.no_grad()
    def evaluate(self, model):
        exact_acc = evaluate_accuracy(model, self.test_loader, self.tokenizer)
        result = {"accuracy": exact_acc}
        log.info(f'result for task "{self.config.name}": {result}')
        return result


class FlanT5GLUETextGenerationRegressionTask(FlanT5GLUETextGenerationTask):
    @torch.no_grad()
    def evaluate(self, model):
        spearman_rho = evaluate_spearman_rho(model, self.test_loader, self.tokenizer)
        result = {"spearman_rho": spearman_rho}
        log.info(f'result for task "{self.config.name}": {result}')
        return result


class FlanT5GLUETextGenerationTaskPool(TaskPool):
    """
    A task pool for FlanT5 GLUE text generation tasks.
    This class manages the tasks and provides methods for loading and evaluating tasks.
    """

    _fabric: L.Fabric = None
    _tokenizer = None

    @property
    def tokenizer(self):
        """
        Returns the tokenizer. If it's not already initialized, it initializes it using the config's tokenizer.
        """
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        return self._tokenizer

    @property
    def fabric(self):
        if self._fabric is not None:
            return self._fabric
        else:
            self._fabric = L.Fabric(devices=1)
            self._fabric.launch()
            return self._fabric

    def load_task(self, task_name_or_config: str | DictConfig):
        """
        Loads a task given a task name or config. If the task name is in `CLASSIFICATION_TASKS`, it creates a `FlanT5GLUETextGenerationClassificationTask`.
        If the task name is in `REGRESSION_TASKS`, it creates a `FlanT5GLUETextGenerationRegressionTask`. Otherwise, it raises a `ValueError`.
        """
        if isinstance(task_name_or_config, str):
            task_config = self.get_task_config(task_name_or_config)
        else:
            task_config = task_name_or_config

        if task_config.name in CLASSIFICATION_TASKS:
            task = FlanT5GLUETextGenerationClassificationTask(task_config)
            task._taskpool = self
            return task
        elif task_config.name in REGRESSION_TASKS:
            task = FlanT5GLUETextGenerationRegressionTask(task_config)
            task._taskpool = self
            return task
        else:
            raise ValueError(f"Unknown task {task_config.name}")

    def evaluate(self, model: T5ForConditionalGeneration):
        if not isinstance(model, T5ForConditionalGeneration):
            log.warning(
                f"Model is not an instance of T5ForConditionalGeneration, but {type(model)}"
            )
        report = {}
        training_params, all_params = count_parameters(model)
        report["model_info"] = {
            "trainable_params": training_params,
            "all_params": all_params,
            "trainable_percentage": training_params / all_params,
        }
        model = self.fabric.setup(model)
        report.update(super().evaluate(model))
        log.info(f"evaluation report: {report}")
        return report
