import functools
import logging
import os
from copy import deepcopy

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    default_data_collator,
)

from fusion_bench.compat.taskpool import TaskPool
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.tasks import BaseTask
from fusion_bench.tasks.flan_t5_text_generation.glue_evaluation import (
    evaluate_accuracy,
    evaluate_spearman_rho,
)
from fusion_bench.tasks.flan_t5_text_generation.glue_load_dataset import (
    load_glue_dataset,
)
from fusion_bench.utils.parameters import count_parameters

log = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CLASSIFICATION_TASKS = [
    "cola",
    "glue-cola",
    "mnli",
    "glue-mnli",
    "mrpc",
    "glue-mrpc",
    "qnli",
    "glue-qnli",
    "qqp",
    "glue-qqp",
    "rte",
    "glue-rte",
    "sst2",
    "glue-sst2",
]
REGRESSION_TASKS = ["stsb", "glue-stsb"]


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


class FlanT5GLUETextGenerationTaskPool(LightningFabricMixin, TaskPool):
    """
    A task pool for FlanT5 GLUE text generation tasks.
    This class manages the tasks and provides methods for loading and evaluating tasks.
    """

    _tokenizer_instance = None

    @property
    def tokenizer(self):
        """
        Returns the tokenizer. If it's not already initialized, it initializes it using the config's tokenizer.
        """
        if self._tokenizer_instance is None:
            self._tokenizer_instance = AutoTokenizer.from_pretrained(
                self.config.tokenizer
            )
        return self._tokenizer_instance

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

    def evaluate(self, model: T5ForConditionalGeneration, name: str = None):
        """
        Evaluate the model on the FlanT5 GLUE text generation tasks.

        Args:
            model (T5ForConditionalGeneration): The model to evaluate.
            name (str, optional): The name of the model. Defaults to None. This is used to identify the model in the report.

        Returns:
            dict: A dictionary containing the evaluation results for each task.
        """
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
        if name is not None:
            report["model_info"]["name"] = name
        model = self.fabric.setup(model)
        report.update(super().evaluate(model))
        log.info(f"evaluation report: {report}")
        return report
