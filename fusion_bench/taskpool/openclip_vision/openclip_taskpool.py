import itertools
import json
import logging
import os
from typing import TYPE_CHECKING, Callable, Dict, Optional, Union

import lightning.fabric
import open_clip
import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, MeanMetric
from torchmetrics.classification.accuracy import MulticlassAccuracy
from tqdm.auto import tqdm

from fusion_bench import BaseTaskPool
from fusion_bench.dataset.clip_dataset import CLIPDataset
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.modelpool.openclip_vision.modelpool import load_classifier_head
from fusion_bench.models.open_clip import (
    ClassificationHead,
    ImageClassifier,
    ImageEncoder,
)
from fusion_bench.models.open_clip.variables_and_paths import OPENCLIP_CACHEDIR
from fusion_bench.utils import count_parameters, instantiate

if TYPE_CHECKING:
    from fusion_bench.modelpool import OpenCLIPVisionModelPool
    from fusion_bench.programs import FabricModelFusionProgram

log = logging.getLogger(__name__)


class OpenCLIPVisionModelTaskPool(
    BaseTaskPool,
    LightningFabricMixin,
):
    _is_setup = False

    _program: "FabricModelFusionProgram"

    processor: Optional[Callable] = None
    test_datasets: Dict[str, CLIPDataset]

    def __init__(
        self,
        test_datasets: Union[DictConfig, Dict[str, Dataset]],
        classification_heads: Union[DictConfig, Dict[str, ClassificationHead]],
        dataloader_kwargs: DictConfig,
        model_name: Optional[str] = None,
        fast_dev_run: bool = False,
        **kwargs,
    ):
        self._test_datasets = test_datasets
        self._classifier_heads = classification_heads
        self._dataloader_kwargs = dataloader_kwargs
        self._model_name = model_name
        self.fast_dev_run = fast_dev_run
        super().__init__(**kwargs)

    def setup(self):
        # setup the processor
        if self._program is not None and self._program.modelpool is not None:
            modelpool: "OpenCLIPVisionModelPool" = self._program.modelpool
            self.processor = modelpool.test_processor
        elif self._model_name is not None:
            _, _, self.processor = open_clip.create_model_and_transforms(
                self._model_name,
                pretrained="openai",
                cache_dir=OPENCLIP_CACHEDIR,
            )
        else:
            raise ValueError("Modelpool or model_name is not set")

        # setup the test datasets
        self.test_datasets = {
            name: instantiate(dataset) if isinstance(dataset, DictConfig) else dataset
            for name, dataset in self._test_datasets.items()
        }
        self.test_datasets = {
            name: CLIPDataset(dataset, self.processor)
            for name, dataset in self.test_datasets.items()
        }
        self.test_dataloaders = {
            name: self.fabric.setup_dataloaders(
                DataLoader(dataset, **self._dataloader_kwargs)
            )
            for name, dataset in self.test_datasets.items()
        }

        # setup classifier heads
        self.classifier_heads = {
            name: load_classifier_head(head).to(self.fabric.device)
            for name, head in self._classifier_heads.items()
        }
        self._is_setup = True

    @torch.no_grad()
    def _evaluate(
        self,
        classifier: ImageClassifier,
        test_loader: DataLoader,
        num_classes: int,
        task_name: str,
    ):
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

        pbar = tqdm(
            test_loader,
            desc=f"Evaluating {task_name}",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in pbar:
            inputs, targets = batch
            logits = classifier(inputs)
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

    def evaluate(self, model: ImageEncoder, **kwargs):
        if not self._is_setup:
            self.setup()

        report = {}
        # collect basic model information
        training_params, all_params = count_parameters(model)
        report["model_info"] = {
            "trainable_params": training_params,
            "all_params": all_params,
            "trainable_percentage": training_params / all_params,
        }

        if not lightning.fabric.is_wrapped(model):
            model = self.fabric.setup_module(model)

        pbar = tqdm(
            self.test_dataloaders.items(),
            desc="Evaluating tasks",
            total=len(self.test_dataloaders),
        )
        for task_name, test_dataloader in pbar:
            classifier = ImageClassifier(model, self.classifier_heads[task_name])
            num_classes = self.classifier_heads[task_name].weight.size(0)
            result = self._evaluate(
                classifier,
                test_dataloader,
                num_classes=num_classes,
                task_name=task_name,
            )
            report[task_name] = result

        # calculate the average accuracy and loss
        if "average" not in report:
            report["average"] = {}
            accuracies = [
                value["accuracy"]
                for key, value in report.items()
                if "accuracy" in value
            ]
            if len(accuracies) > 0:
                average_accuracy = sum(accuracies) / len(accuracies)
                report["average"]["accuracy"] = average_accuracy
            losses = [value["loss"] for key, value in report.items() if "loss" in value]
            if len(losses) > 0:
                average_loss = sum(losses) / len(losses)
                report["average"]["loss"] = average_loss

        log.info(f"Evaluation Result: {report}")
        if self.fabric.is_global_zero and len(self.fabric._loggers) > 0:
            with open(os.path.join(self.log_dir, "report.json"), "w") as fp:
                json.dump(report, fp)
        return report
