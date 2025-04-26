import functools
import itertools
import logging
from copy import deepcopy
from typing import Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MeanMetric
from tqdm.autonotebook import tqdm
from transformers import (
    GPT2ForSequenceClassification,
    GPT2Model,
    GPT2Tokenizer,
    default_data_collator,
)
from typing_extensions import override

from fusion_bench.dataset.gpt2_glue import TokenizedGLUE
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.taskpool import BaseTaskPool
from fusion_bench.utils import instantiate

log = logging.getLogger(__name__)

tokenizer: GPT2Tokenizer = None


@functools.cache
def load_gpt2_dataset(name: str, split: Optional[str] = None):
    global tokenizer
    dataset = TokenizedGLUE(tokenizer=tokenizer).load_dataset(name)
    if split is not None:
        dataset = dataset[split]
    return dataset


class GPT2TextClassificationTaskPool(BaseTaskPool, LightningFabricMixin):
    """
    A task pool for GPT2 text classification tasks.
    This class manages the tasks and provides methods for loading test dataset and evaluation.
    """

    _config_mapping = BaseTaskPool._config_mapping | {
        "_test_datasets": "test_datasets",
        "_tokenizer": "tokenizer",
        "dataloader_kwargs": "dataloader_kwargs",
        "fast_dev_run": "fast_dev_run",
    }

    def __init__(
        self,
        test_datasets: DictConfig,
        tokenizer: DictConfig,
        dataloader_kwargs: DictConfig,
        fast_dev_run: bool,
        **kwargs,
    ):
        self._test_datasets = test_datasets
        self._tokenizer = tokenizer
        self.dataloader_kwargs = dataloader_kwargs
        self.fast_dev_run = fast_dev_run
        super().__init__(**kwargs)

        self.setup()

    def setup(self):
        global tokenizer
        self.tokenizer = tokenizer = instantiate(self._tokenizer)

    def get_classifier(
        self, task_name: str, model: GPT2Model
    ) -> GPT2ForSequenceClassification:
        modelpool = self._program.modelpool
        classifier = modelpool.load_classifier(task_name)
        classifier.transformer = deepcopy(model)
        return classifier

    @torch.no_grad()
    def evaluate_single_task(
        self,
        task_name: str,
        model: GPT2Model,
        test_loader: DataLoader,
    ):
        loss_metric = MeanMetric()
        # load classifier and replace the backbone with the passed model
        model: GPT2ForSequenceClassification = self.get_classifier(task_name, model)
        accuracy = Accuracy("multiclass", num_classes=model.num_labels)
        model = self.fabric.setup(model)

        if self.config.get("fast_dev_run", False):
            log.info("Running under fast_dev_run mode, evaluating on a single batch.")
            test_loader = itertools.islice(test_loader, 1)
        else:
            test_loader = test_loader

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

            accuracy(logits.detach().cpu(), labels.detach().cpu())
            loss_metric.update(loss.detach().cpu())
            pbar.set_postfix(
                {
                    "accuracy": accuracy.compute().item(),
                    "loss": loss_metric.compute().item(),
                }
            )

        acc = accuracy.compute().item()
        loss = loss_metric.compute().item()
        results = {"accuracy": acc, "loss": loss}
        log.info(f"Results for task {task_name}: {results}")
        return results

    def get_test_dataloader(self, task_name: str):
        dataset = instantiate(self._test_datasets[task_name])
        dataloader_kwargs = {
            "shuffle": False,
        }
        dataloader_kwargs.update(self.dataloader_kwargs)
        dataloader = DataLoader(
            dataset, collate_fn=default_data_collator, **dataloader_kwargs
        )
        if self.fabric is not None:
            dataloader = self.fabric.setup_dataloaders(dataloader)
        return dataloader

    @override
    def evaluate(self, model: GPT2Model, name: str = None):
        """Evaluate the model on the test datasets.

        Args:
            model (GPT2Model): The model to evaluate.
            name (str, optional): The name of the model. Defaults to None. This is used to identify the model in the report.

        Returns:
            dict: A dictionary containing the evaluation results for each task.
        """
        report = {}
        if name is not None:
            report["name"] = name
        for task_name in (pbar := tqdm(self._test_datasets, desc="Evaluating tasks")):
            pbar.set_description(f"Evaluating task {task_name}")
            dataloader = self.get_test_dataloader(task_name)
            result = self.evaluate_single_task(task_name, model, dataloader)
            report[task_name] = result

        # calculate the average accuracy and loss
        if "average" not in report:
            report["average"] = {}
            accuracies = [
                value["accuracy"]
                for key, value in report.items()
                if isinstance(value, dict) and "accuracy" in value
            ]
            if len(accuracies) > 0:
                average_accuracy = sum(accuracies) / len(accuracies)
                report["average"]["accuracy"] = average_accuracy
            losses = [value["loss"] for key, value in report.items() if "loss" in value]
            if len(losses) > 0:
                average_loss = sum(losses) / len(losses)
                report["average"]["loss"] = average_loss

        log.info(f"Evaluation Result: {report}")
        return report
