import functools
import itertools
import logging
from copy import deepcopy

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MeanMetric
from tqdm.autonotebook import tqdm
from transformers import (
    GPT2ForSequenceClassification,
    GPT2Model,
    GPT2Tokenizer,
    default_data_collator,
)

import fusion_bench
from fusion_bench.dataset.gpt2_glue import TokenizedGLUE
from fusion_bench.taskpool import TaskPool
from fusion_bench.tasks import BaseTask

log = logging.getLogger(__name__)


class GPT2ClassificationTask(BaseTask):
    _taskpool: "GPT2TextClassificationTaskPool" = None

    def __init__(
        self, task_config: DictConfig, fabric: L.Fabric, tokenizer: GPT2Tokenizer
    ):
        super().__init__(task_config)
        self._fabric = fabric
        self._tokenizer = tokenizer

    @property
    def num_classes(self):
        return len(self.test_dataset.unique("label"))

    @functools.cached_property
    def dataset(self):
        log.info('Loading dataset: "{}"'.format(self.config.dataset.name))
        dataset = TokenizedGLUE(tokenizer=self._tokenizer).load_dataset(
            self.config.dataset.name
        )
        return dataset

    @property
    def test_dataset(self):
        return self.dataset[self.config.dataset.split]

    @property
    def test_loader(self):
        test_dataset = self.test_dataset
        loader = DataLoader(
            test_dataset,
            collate_fn=default_data_collator,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
        )
        if self._fabric is not None:
            loader = self._fabric.setup_dataloaders(loader)
        return loader

    def get_classifier(self, model: GPT2Model) -> GPT2ForSequenceClassification:
        modelpool = self._taskpool._modelpool
        classifier = modelpool.load_classifier(self.config.name)
        classifier.transformer = deepcopy(model)
        return classifier

    @torch.no_grad()
    def evaluate(self, model: GPT2Model):
        accuracy = Accuracy("multiclass", num_classes=self.num_classes)
        loss_metric = MeanMetric()
        model: GPT2ForSequenceClassification = self.get_classifier(model)
        model = self._fabric.setup(model)

        if self.config.get("fast_dev_run", False):
            log.info("Running under fast_dev_run mode, evaluating on a single batch.")
            test_loader = itertools.islice(self.test_loader, 1)
        else:
            test_loader = self.test_loader

        for batch in (
            pbar := tqdm(
                test_loader, desc="Evaluating", leave=False, dynamic_ncols=True
            )
        ):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = F.cross_entropy(logits, labels)

            acc = accuracy(logits.detach().cpu(), labels.detach().cpu())
            loss_metric.update(loss.detach().cpu())
            pbar.set_postfix({"accuracy": acc.item(), "loss": loss.item()})

        acc = accuracy.compute().item()
        loss = loss_metric.compute().item()
        results = {"accuracy": acc, "loss": loss}
        log.info(f"Results for task {self.config.name}: {results}")
        return results


class GPT2TextClassificationTaskPool(TaskPool):
    """
    A task pool for GPT2 text classification tasks.
    This class manages the tasks and provides methods for loading test dataset and evaluation.
    """

    _fabric: L.Fabric = None
    _tokenizer: GPT2Tokenizer = None
    _modelpool: "fusion_bench.modelpool.HuggingFaceGPT2ClassificationPool" = None

    @property
    def fabric(self):
        if self._fabric is not None:
            return self._fabric
        else:
            self._fabric = L.Fabric(devices=1)
            self._fabric.launch()
            return self._fabric

    @property
    def tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        else:
            raise ValueError("Tokenizer not set")

    def prepare_dataset_config(self, dataset_config: DictConfig):
        """
        Set default values for dataset configuration.
        """
        if not hasattr(dataset_config, "type"):
            with open_dict(dataset_config):
                dataset_config["type"] = self.config.dataset_type
        return dataset_config

    def prepare_task_config(self, task_config: DictConfig):
        """
        Set default values for task configuration.
        """
        for key in ["num_workers", "batch_size", "fast_dev_run"]:
            if not hasattr(task_config, key):
                with open_dict(task_config):
                    task_config[key] = self.config[key]
        return task_config

    def load_task(self, task_name_or_config: str | DictConfig):
        """
        Loads a task given a task name or config. It prepares the task configuration and loads the task from it.
        """
        if isinstance(task_name_or_config, str):
            task_config = self.get_task_config(task_name_or_config)
        else:
            task_config = task_name_or_config
        task_config = self.prepare_task_config(task_config)

        # load the task from the configuration
        task = GPT2ClassificationTask(task_config, self.fabric, self.tokenizer)
        task._fabric = self._fabric
        task._tokenizer = self._tokenizer
        task._taskpool = self

        return task
