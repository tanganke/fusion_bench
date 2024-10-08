import functools
import itertools
import json
import logging
import os
from typing import Any, Callable, Dict, List, Union, cast

import lightning as L
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, MeanMetric
from torchmetrics.classification.accuracy import MulticlassAccuracy
from tqdm.autonotebook import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
from transformers.models.clip.modeling_clip import CLIPVisionTransformer

from fusion_bench.dataset import CLIPDataset
from fusion_bench.mixins import LightningFabricMixin, YAMLSerializationMixin
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.taskpool import BaseTaskPool
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.utils import instantiate
from fusion_bench.utils.parameters import count_parameters

os.environ["TOKENIZERS_PARALLELISM"] = "false"

log = logging.getLogger(__name__)


def raw_image_collate_fn(batch):
    images, labels = tuple(zip(*batch))
    labels = torch.as_tensor(labels)
    return images, labels


class CLIPVisionModelTaskPool(LightningFabricMixin, BaseTaskPool):
    """
    This class is used to define the image classification task for CLIP models.
    """

    _is_setup = False
    _config_mapping = BaseTaskPool._config_mapping | {
        "_test_datasets": "test_datasets",
        "_processor": "processor",
        "_data_processor": "data_processor",
        "_clip_model": "clip_model",
        "_dataloader_kwargs": "dataloader_kwargs",
        "fast_dev_run": "fast_dev_run",
        "_version_": "_version",
    }

    def __init__(
        self,
        test_datasets: Union[DictConfig, Dict[str, Dataset]],
        *,
        processor: Union[DictConfig, CLIPProcessor],
        data_processor: Union[DictConfig, CLIPProcessor],
        clip_model: Union[DictConfig, CLIPModel],
        dataloader_kwargs: DictConfig = None,
        fast_dev_run: bool = False,
        _version_: str = None,
        **kwargs,
    ):
        super().__init__()
        self._test_datasets = test_datasets
        self._processor = processor
        self._data_processor = data_processor
        self._clip_model = clip_model
        self._dataloader_kwargs = dataloader_kwargs or {}
        self.fast_dev_run = fast_dev_run
        self._version_ = _version_
        log.warning(f"Unrecognized arguments: {list(kwargs.keys())}")

    def setup(self):
        # setup processor and clip model
        self.processor = (
            instantiate(self._processor)
            if isinstance(self._processor, DictConfig)
            else self._processor
        )
        self.data_processor = (
            instantiate(self._data_processor)
            if isinstance(self._data_processor, DictConfig)
            else self._data_processor
        )
        self.clip_model = (
            instantiate(self._clip_model)
            if isinstance(self._clip_model, DictConfig)
            else self._clip_model
        )
        self.clip_model = self.fabric.to_device(self.clip_model)
        self.clip_model.requires_grad_(False)
        self.clip_model.eval()

        # Load the test datasets
        self.test_datasets = {
            name: instantiate(dataset) if isinstance(dataset, DictConfig) else dataset
            for name, dataset in self._test_datasets.items()
        }
        self.test_datasets = {
            name: CLIPDataset(dataset, self.data_processor)
            for name, dataset in self.test_datasets.items()
        }
        # Setup the dataloaders
        self.test_dataloaders = {
            name: DataLoader(
                dataset,
                **self._dataloader_kwargs,
                collate_fn=(
                    raw_image_collate_fn if self.data_processor is None else None
                ),
            )
            for name, dataset in self.test_datasets.items()
        }
        self.test_dataloaders = {
            name: self.fabric.setup_dataloaders(dataloader)
            for name, dataloader in self.test_dataloaders.items()
        }

        self._is_setup = True

    @torch.no_grad()
    def _evaluate(self, classifier: nn.Module, test_loader, num_classes: int):
        accuracy: MulticlassAccuracy = Accuracy(
            task="multiclass", num_classes=num_classes
        )
        classifier.eval()
        loss_metric = MeanMetric()
        # if fast_dev_run is set, we only evaluate on a batch of the data
        if self.fast_dev_run:
            log.info("Running under fast_dev_run mode, evaluating on a single batch.")
            test_loader = itertools.islice(test_loader, 1)
        else:
            test_loader = test_loader

        for batch in (
            pbar := tqdm(
                test_loader, desc="Evaluating", leave=False, dynamic_ncols=True
            )
        ):
            inputs, targets = batch
            logits: Tensor = classifier(inputs)

            loss = F.cross_entropy(logits, targets)
            loss_metric.update(loss.detach().cpu())
            acc = accuracy(logits.detach().cpu(), targets.detach().cpu())
            pbar.set_postfix(
                {
                    "accuracy": accuracy.compute().item(),
                    "loss": loss_metric.compute().item(),
                }
            )

        acc = accuracy.compute().item()
        loss = loss_metric.compute().item()
        results = {"accuracy": acc, "loss": loss}
        return results

    def evaluate(self, model: Union[CLIPVisionModel, CLIPVisionTransformer]):
        """
        Evaluate the model on the image classification task.
        """
        if not self._is_setup:
            self.setup()

        report = {}
        # CLIPVisionModel works the same with CLIPVisonTransformer, so we can use it directly
        self.clip_model.vision_model = model
        classifier = HFCLIPClassifier(self.clip_model, processor=self.processor)
        classifier = self.fabric.to_device(classifier)
        # collect basic model information
        training_params, all_params = count_parameters(model)
        report["model_info"] = {
            "trainable_params": training_params,
            "all_params": all_params,
            "trainable_percentage": training_params / all_params,
        }
        for task_name, test_dataloader in tqdm(
            self.test_dataloaders.items(),
            desc="Evaluating tasks",
            total=len(self.test_dataloaders),
        ):
            classnames, templates = get_classnames_and_templates(task_name)
            classifier.set_classification_task(classnames, templates)
            result = self._evaluate(
                classifier, test_dataloader, num_classes=len(classnames)
            )
            report[task_name] = result
        log.info(f"Evaluation Result: {report}")
        if self.fabric.is_global_zero and len(self.fabric._loggers) > 0:
            with open(os.path.join(self.log_dir, "report.json"), "w") as fp:
                json.dump(report, fp)
        return report
