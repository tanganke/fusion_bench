import functools
import logging
import os
from copy import deepcopy
from typing import Dict, List, Optional

import lightning.fabric
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    default_data_collator,
)

from fusion_bench import BaseTaskPool
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
from fusion_bench.utils import instantiate

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


class FlanT5GLUETextGenerationTask:
    _taskpool: "FlanT5GLUETextGenerationTaskPool" = None

    def __init__(
        self,
        name: str,
        split: str,
        taskpool_ref: Optional["FlanT5GLUETextGenerationTaskPool"] = None,
    ):
        self.name = name
        self.split = split
        self._taskpool = taskpool_ref

    @property
    def taskpool(self):
        if self._taskpool is not None:
            return self._taskpool
        else:
            raise ValueError("Taskpool not set, set this after initialization.")

    @property
    def fabric(self):
        return self.taskpool.fabric

    @property
    def tokenizer(self):
        return self.taskpool.tokenizer

    @functools.cached_property
    def dataset(self):
        log.info(f'Loading dataset: "{self.name}"')
        dataset = load_glue_dataset(self.name, self.tokenizer, self.taskpool.cache_dir)
        return dataset

    @functools.cached_property
    def test_dataset(self):
        return self.dataset[self.split]

    @property
    def test_loader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.taskpool.batch_size,
            num_workers=self.taskpool.num_workers,
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
        log.info(f'result for task "{self.name}": {result}')
        return result


class FlanT5GLUETextGenerationRegressionTask(FlanT5GLUETextGenerationTask):
    @torch.no_grad()
    def evaluate(self, model):
        spearman_rho = evaluate_spearman_rho(model, self.test_loader, self.tokenizer)
        result = {"spearman_rho": spearman_rho}
        log.info(f'result for task "{self.name}": {result}')
        return result


class FlanT5GLUETextGenerationTaskPool(
    BaseTaskPool,
    LightningFabricMixin,
):
    """
    A task pool for FlanT5 GLUE text generation tasks.
    This class manages the tasks and provides methods for loading and evaluating tasks.
    """

    _tokenizer_instance = None

    _config_mapping = BaseTaskPool._config_mapping | {
        "_tokenizer": "tokenizer",
        "cache_dir": "cache_dir",
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        "fast_dev_run": "fast_dev_run",
    }

    def __init__(
        self,
        tasks: List[DictConfig],
        tokenizer: str = "google/flan-t5-base",
        cache_dir: str = "outputs/cache",
        batch_size: int = 32,
        num_workers: int = 4,
        fast_dev_run: bool = False,
        **kwargs,
    ):
        self._tasks = tasks
        """list of task configs"""
        self._tokenizer = tokenizer
        """path to the tokenizer"""
        self.cache_dir = cache_dir
        """path to the cache directory"""
        self.batch_size = batch_size
        """batch size"""
        self.num_workers = num_workers
        """number of workers"""
        self.fast_dev_run = fast_dev_run
        """whether to run in fast dev run mode (debug mode)"""
        super().__init__(**kwargs)

    @property
    def task_names(self):
        """
        Return a list of all task names in the task pool.

        Returns:
            List[str]: A list of all task names.
        """
        return list(self._tasks.keys())

    @property
    def tokenizer(self):
        """
        Returns the tokenizer. If it's not already initialized, it initializes it using the config's tokenizer.
        """
        if self._tokenizer_instance is None:
            log.info(f"Initializing tokenizer from: {self._tokenizer}")
            self._tokenizer_instance = AutoTokenizer.from_pretrained(self._tokenizer)
        return self._tokenizer_instance

    def load_task(
        self, task_name_or_config: str | DictConfig
    ) -> "FlanT5GLUETextGenerationTask":
        """
        Loads a task given a task name or config. If the task name is in `CLASSIFICATION_TASKS`, it creates a `FlanT5GLUETextGenerationClassificationTask`.
        If the task name is in `REGRESSION_TASKS`, it creates a `FlanT5GLUETextGenerationRegressionTask`. Otherwise, it raises a `ValueError`.
        """
        if isinstance(task_name_or_config, str):
            task_config = self._tasks[task_name_or_config]
        else:
            task_config = task_name_or_config

        task = instantiate(task_config, taskpool_ref=self)
        return task

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
        # collect basic model information
        training_params, all_params = count_parameters(model)
        report["model_info"] = {
            "trainable_params": training_params,
            "all_params": all_params,
            "trainable_percentage": training_params / all_params,
        }
        if name is not None:
            report["model_info"]["name"] = name
        model = self.fabric.setup(model)

        if not lightning.fabric.is_wrapped(model):
            model = self.fabric.setup(model)
        # evaluate each task
        for task_name in tqdm(self.task_names, desc="Evaluating tasks"):
            task = self.load_task(task_name)
            result = task.evaluate(model)
            report[task_name] = result
        log.info(f"evaluation report: {report}")
        return report
