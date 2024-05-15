import logging
import os
from functools import cached_property
from typing import Callable, List, cast

import lightning as L
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel

from ..dataset import load_dataset_from_config
from ..models.hf_clip import HFCLIPClassifier
from ..tasks.clip_classification import get_classnames_and_templates
from ..tasks.image_classification import ImageClassificationTask
from .base_pool import TaskPool

os.environ["TOKENIZERS_PARALLELISM"] = "false"

log = logging.getLogger(__name__)


class CLIPImageClassificationTask(ImageClassificationTask):
    _fabric: L.Fabric = None
    _clip_processor: CLIPProcessor = None
    #
    _taskpool: "CLIPImageClassificationTaskPool" = None

    classnames: List[str] = []
    templates: List[Callable[[str], str]] = []

    def __init__(self, task_config: DictConfig):
        self.config = task_config

        self.classnames, self.templates = get_classnames_and_templates(
            self.config["dataset"].name
        )

    @cached_property
    def test_dataset(self):
        dataset_config = self.config["dataset"]
        dataset_config = self._taskpool.prepare_dataset_config(dataset_config)
        log.info(f"Loading test dataset: {dataset_config.name}")
        return load_dataset_from_config(self.config["dataset"])

    @property
    def num_classes(self):
        return len(self.classnames)

    @property
    def test_loader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
        )

    def evaluate(self, clip_model: CLIPModel):
        """
        Evaluate the model on the image classification task.
        """
        classifier = HFCLIPClassifier(
            clip_model=clip_model, processor=self._clip_processor
        )
        classifier.set_classification_task(self.classnames, self.templates)
        super().evaluate(classifier)


class CLIPImageClassificationTaskPool(TaskPool):
    # CLIP forward model and processor
    _clip_model: CLIPModel = None
    _clip_processor: CLIPProcessor = None

    def __init__(self, taskpool_config: DictConfig):
        super().__init__(taskpool_config)

    def prepare_dataset_config(self, dataset_config: DictConfig):
        if not hasattr(dataset_config, "type"):
            with open_dict(dataset_config):
                dataset_config["type"] = self.config.dataset_type
        return dataset_config

    @property
    def clip_model(self):
        if self._clip_model is None:
            self._clip_model = CLIPModel.from_pretrained(self.config["clip_model"])
        return self._clip_model

    @property
    def clip_processor(self):
        if self._clip_processor is None:
            self._clip_processor = CLIPProcessor.from_pretrained(
                self.config["clip_model"]
            )
        return self._clip_processor

    def load_task(self, task_name_or_config: str | DictConfig):
        if isinstance(task_name_or_config, str):
            task_config = self.get_task_config(task_name_or_config)
        else:
            task_config = task_name_or_config

        # load the task from the configuration
        task = CLIPImageClassificationTask(task_config)
        task._taskpool = self
        task._clip_processor = self.clip_processor
        return task

    def evaluate(self, model: CLIPVisionModel):
        """
        Evaluate the model on the image classification task.
        """
        self.clip_model.vision_model = model
        report = {}
        for task_name in self.task_names:
            task = self.load_task(task_name)
            result = task.evaluate(self.clip_model)
            report[task_name] = result
        return report
