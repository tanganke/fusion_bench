import itertools
import json
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    TypeVar,
    Union,
    override,
)

import lightning as L
import torch
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MeanMetric
from tqdm.auto import tqdm

from fusion_bench import (
    BaseTaskPool,
    LightningFabricMixin,
    RuntimeConstants,
    auto_register_config,
    get_rankzero_logger,
)
from fusion_bench.dataset import CLIPDataset
from fusion_bench.modelpool.resnet_for_image_classification import (
    ResNetForImageClassificationPool,
    load_torchvision_resnet,
    load_transformers_resnet,
)
from fusion_bench.tasks.clip_classification import get_classnames, get_num_classes
from fusion_bench.utils import count_parameters

if TYPE_CHECKING:
    from torchvision.models import ResNet as TorchVisionResNet
    from transformers import ResNetForImageClassification

log = get_rankzero_logger(__name__)

__all__ = ["ResNetForImageClassificationTaskPool"]


@auto_register_config
class ResNetForImageClassificationTaskPool(
    BaseTaskPool,
    LightningFabricMixin,
    ResNetForImageClassificationPool,
):

    _is_setup = False

    def __init__(
        self,
        type: str,
        test_datasets: DictConfig,
        dataloader_kwargs: DictConfig,
        processor_config_path: str,
        **kwargs,
    ):
        if type == "transformers":
            super().__init__(
                models=DictConfig(
                    {"_pretrained_": {"config_path": processor_config_path}}
                ),
                type=type,
                test_datasets=test_datasets,
                **kwargs,
            )
        elif type == "torchvision":
            super().__init__(type=type, test_datasets=test_datasets, **kwargs)
        else:
            raise ValueError(f"Unknown ResNet type: {type}")

    def setup(self):
        processor = self.load_processor(stage="test")

        # Load test datasets
        test_datasets = {
            ds_name: CLIPDataset(self.load_test_dataset(ds_name), processor=processor)
            for ds_name in self._test_datasets
        }
        self.test_dataloaders = {
            ds_name: self.fabric.setup_dataloaders(
                self.get_dataloader(ds, stage="test")
            )
            for ds_name, ds in test_datasets.items()
        }

    def _evaluate(
        self,
        classifier,
        test_loader,
        num_classes: int,
        task_name: str = None,
    ):
        classifier.eval()
        accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        loss_metric = MeanMetric()
        if RuntimeConstants.debug:
            log.info("Running under fast_dev_run mode, evaluating on a single batch.")
            test_loader = itertools.islice(test_loader, 1)
        else:
            test_loader = test_loader

        pbar = tqdm(
            test_loader,
            desc=f"Evaluating {task_name}" if task_name is not None else "Evaluating",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in pbar:
            inputs, targets = batch
            outputs = classifier(inputs)
            logits: Tensor = outputs["logits"]
            if logits.device != targets.device:
                targets = targets.to(logits.device)

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

    def evaluate(
        self,
        model: Union["ResNetForImageClassification", "TorchVisionResNet"],
        name: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        assert isinstance(
            model, nn.Module
        ), f"Expected model to be an instance of nn.Module, but got {type(model)}"

        if not self._is_setup:
            self.setup()

        classifier = self.fabric.to_device(model)
        classifier.eval()
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

        # evaluate on each task
        pbar = tqdm(
            self.test_dataloaders.items(),
            desc="Evaluating tasks",
            total=len(self.test_dataloaders),
        )
        for task_name, test_dataloader in pbar:
            num_classes = get_num_classes(task_name)
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
            save_path = os.path.join(self.log_dir, "report.json")
            for version in itertools.count(1):
                if not os.path.exists(save_path):
                    break
                # if the file already exists, increment the version to avoid overwriting
                save_path = os.path.join(self.log_dir, f"report_{version}.json")
            with open(save_path, "w") as fp:
                json.dump(report, fp)
            log.info(f"Evaluation report saved to {save_path}")
        return report

    def get_dataloader(self, dataset, stage: str):
        """Create a DataLoader for the specified dataset and training stage.

        Constructs a PyTorch DataLoader with stage-appropriate configurations:
        - Training stage: shuffling enabled by default
        - Validation/test stages: shuffling disabled by default

        Args:
            dataset: The dataset to wrap in a DataLoader.
            stage (str): Training stage, must be one of "train", "val", or "test".
                Determines default shuffling behavior.

        Returns:
            DataLoader: Configured DataLoader for the given dataset and stage.
        """
        assert stage in ["train", "val", "test"], f"Invalid stage: {stage}"
        dataloader_kwargs = dict(self.dataloader_kwargs)
        if "shuffle" not in dataloader_kwargs:
            dataloader_kwargs["shuffle"] = stage == "train"
        return DataLoader(dataset, **dataloader_kwargs)
