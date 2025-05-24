import itertools
import json
import logging
import os
from pathlib import Path
from typing import (  # noqa: F401
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.hooks import RemovableHandle
from torchmetrics import Accuracy, MeanMetric
from torchmetrics.classification.accuracy import MulticlassAccuracy
from tqdm.autonotebook import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
from transformers.models.clip.modeling_clip import CLIPVisionTransformer

from fusion_bench.dataset import CLIPDataset
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.taskpool import BaseTaskPool
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.utils import count_parameters, instantiate

if TYPE_CHECKING:
    from fusion_bench.models.surgery.surgerymodelwrapper import SurgeryModelWrapper

# disable tokenizers parallelism by default to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

log = logging.getLogger(__name__)


def raw_image_collate_fn(batch):
    images, labels = tuple(zip(*batch))
    labels = torch.as_tensor(labels)
    return images, labels


class LayerWiseFeatureSaver:
    def __init__(
        self,
        save_path: Path,
        first_token_only: bool = True,
        max_num: Optional[int] = None,
    ):
        self.save_path = save_path
        self.first_token_only = first_token_only
        self.max_num = max_num
        self.features = []

    def __call__(self, module, input, output: Tuple[Tensor]):
        features = output[0].detach().cpu()
        if self.first_token_only:
            features = features[:, 0]
        if self.max_num is not None and self.max_num > 0:
            if len(self.features) > self.max_num:
                return
            elif features.size(0) + len(self.features) > self.max_num:
                self.features.append(features[: self.max_num - len(self.features)])
            else:
                self.features.append(features)
        else:
            self.features.append(features)

    def save_features(self):
        features = torch.cat(self.features, dim=0)
        if self.save_path is not None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving features to {self.save_path}")
            torch.save(features, self.save_path)


class CLIPVisionModelTaskPool(
    BaseTaskPool,
    LightningFabricMixin,
):
    """
    This class is used to define the image classification task for CLIP models.

    Attributes:
        test_datasets (Union[DictConfig, Dict[str, Dataset]]): The test datasets to evaluate the model on.
        processor (Union[DictConfig, CLIPProcessor]): The processor used for preprocessing the input data.
        data_processor (Union[DictConfig, CLIPProcessor]): The data processor used for processing the input data.
        clip_model (Union[DictConfig, CLIPModel]): The CLIP model used for evaluation.
        dataloader_kwargs (DictConfig): Keyword arguments for the data loader.
        layer_wise_feature_save_path (Optional[str]): Path to save the layer-wise features.
        layer_wise_feature_first_token_only (bool): Boolean indicating whether to save only the first token of the features.
        layer_wise_feature_max_num (Optional[int]): Maximum number of features to save.
        fast_dev_run (bool): Boolean indicating whether to run in fast development mode.
    """

    _is_setup = False

    # hooks and handles for saving layer-wise features
    _layer_wise_feature_save_hooks: Dict[int, LayerWiseFeatureSaver] = {}
    _layer_wise_feature_save_hook_handles: Dict[int, RemovableHandle] = {}

    _config_mapping = BaseTaskPool._config_mapping | {
        "_test_datasets": "test_datasets",
        "_processor": "processor",
        "_data_processor": "data_processor",
        "_clip_model": "clip_model",
        "_dataloader_kwargs": "dataloader_kwargs",
        "_layer_wise_feature_save_path": "layer_wise_feature_save_path",
        "fast_dev_run": "fast_dev_run",
    }

    def __init__(
        self,
        test_datasets: Union[DictConfig, Dict[str, Dataset]],
        *,
        processor: Union[DictConfig, CLIPProcessor],
        data_processor: Union[DictConfig, CLIPProcessor],
        clip_model: Union[DictConfig, CLIPModel],
        dataloader_kwargs: DictConfig = None,
        layer_wise_feature_save_path: Optional[str] = None,
        layer_wise_feature_first_token_only: bool = True,
        layer_wise_feature_max_num: Optional[int] = None,
        fast_dev_run: bool = False,
        **kwargs,
    ):
        """
        Initialize the CLIPVisionModelTaskPool.
        """
        self._test_datasets = test_datasets
        self._processor = processor
        self._data_processor = data_processor
        self._clip_model = clip_model
        self._dataloader_kwargs = dataloader_kwargs or {}

        # layer-wise feature saving
        self._layer_wise_feature_save_path = layer_wise_feature_save_path
        self.layer_wise_feature_save_path = (
            Path(layer_wise_feature_save_path)
            if layer_wise_feature_save_path is not None
            else None
        )
        self.layer_wise_feature_first_token_only = layer_wise_feature_first_token_only
        self.layer_wise_feature_max_num = layer_wise_feature_max_num

        self.fast_dev_run = fast_dev_run
        super().__init__(**kwargs)

    def setup(self):
        """
        Set up the processor, data processor, CLIP model, test datasets, and data loaders.
        """
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
    def _evaluate(
        self,
        classifier: HFCLIPClassifier,
        test_loader: DataLoader,
        num_classes: int,
        task_name: str = None,
    ):
        """
        Evaluate the classifier on the test dataset (single-task evaluation).

        Args:
            classifier (HFCLIPClassifier): The classifier to evaluate.
            test_loader (DataLoader): The data loader for the test dataset.
            num_classes (int): The number of classes in the classification task.
            task_name (str): The name of the task.

        Returns:
            Dict[str, float]: A dictionary containing the accuracy and loss of the classifier on the test dataset.
        """
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
            outputs = classifier(
                inputs,
                return_image_embeds=True,
                return_dict=True,
                task_name=task_name,
            )
            logits: Tensor = outputs["logits"]

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
        model: Union[CLIPVisionModel, CLIPVisionTransformer],
        name=None,
        **kwargs,
    ):
        """
        Evaluate the model on the image classification task.

        Args:
            model (Union[CLIPVisionModel, CLIPVisionTransformer]): The model to evaluate.
            name (Optional[str]): The name of the model. This will be logged into the report if not None.

        Returns:
            Dict[str, Any]: A dictionary containing the evaluation results for each task.
        """
        if not self._is_setup:
            self.setup()

        report = {}
        # CLIPVisionModel works the same with CLIPVisonTransformer, so we can use it directly
        if hasattr(model, "is_surgery_model") and model.is_surgery_model:
            log.info("running evaluation on a surgery model.")
            model: "SurgeryModelWrapper" = model
            self.clip_model.vision_model = model
        else:
            # replace the vision encoder with the model
            self.clip_model.vision_model = model
        classifier = HFCLIPClassifier(
            self.clip_model,
            processor=self.processor,
        )
        classifier = cast(HFCLIPClassifier, self.fabric.to_device(classifier))
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
            classnames, templates = get_classnames_and_templates(task_name)
            self.on_task_evaluation_begin(classifier, task_name)
            classifier.set_classification_task(classnames, templates)
            result = self._evaluate(
                classifier,
                test_dataloader,
                num_classes=len(classnames),
                task_name=task_name,
            )
            report[task_name] = result
            self.on_task_evaluation_end()

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

    def on_task_evaluation_begin(self, classifier: HFCLIPClassifier, task_name: str):
        """
        Called at the beginning of task evaluation to set up hooks for saving layer-wise features.

        Args:
            classifier (HFCLIPClassifier): The classifier being evaluated.
            task_name (str): The name of the task being evaluated.
        """
        if self.layer_wise_feature_save_path is not None:
            # setup hooks for saving layer-wise features
            assert isinstance(
                classifier.clip_model.vision_model,
                (CLIPVisionTransformer, CLIPVisionModel),
            ), "Vision model is expected to be a CLIPVisionTransformer"
            vision_model = classifier.clip_model.vision_model
            if isinstance(vision_model, CLIPVisionModel):
                vision_model = vision_model.vision_model
                # assign forward hooks for each layer
            for i, layer in enumerate(vision_model.encoder.layers):
                self._layer_wise_feature_save_hooks[i] = LayerWiseFeatureSaver(
                    self.layer_wise_feature_save_path / task_name / f"layer_{i}.pth",
                    first_token_only=self.layer_wise_feature_first_token_only,
                    max_num=self.layer_wise_feature_max_num,
                )
                self._layer_wise_feature_save_hook_handles[i] = (
                    layer.register_forward_hook(self._layer_wise_feature_save_hooks[i])
                )

    def on_task_evaluation_end(self):
        """
        Called at the end of task evaluation to save features and remove hooks.
        """
        if self.layer_wise_feature_save_path is not None:
            # save features and remove hooks after evaluation
            for i, hook in self._layer_wise_feature_save_hooks.items():
                hook.save_features()
                self._layer_wise_feature_save_hook_handles[i].remove()
