import functools
import json
import logging
import os
from copy import deepcopy
from functools import cached_property
from typing import Callable, List, cast

import lightning as L
import torch
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel

from fusion_bench.dataset import CLIPDataset, load_dataset_from_config
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.taskpool import TaskPool
from fusion_bench.tasks.classification import ClassificationTask
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.utils.parameters import count_parameters

os.environ["TOKENIZERS_PARALLELISM"] = "false"

log = logging.getLogger(__name__)


@functools.cache
def load_dataset_from_config_cached(dataset_config: DictConfig):
    return load_dataset_from_config(dataset_config)


class CLIPImageClassificationTask(ClassificationTask):
    """
    This class is used to define the image classification task for CLIP models.
    """

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
        """
        Load the test dataset for the task.
        This method is cached, so the dataset is loaded only once.
        """
        dataset_config = self.config["dataset"]
        dataset_config = self._taskpool.prepare_dataset_config(dataset_config)
        log.info(f"Loading test dataset: {dataset_config.name}")
        dataset = load_dataset_from_config_cached(dataset_config)
        dataset = CLIPDataset(dataset, self._clip_processor)
        return dataset

    @property
    def num_classes(self):
        return len(self.classnames)

    @property
    def test_loader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
        )
        if self._fabric is not None:
            loader = self._fabric.setup_dataloaders(loader)
        return loader

    def evaluate(self, clip_model: CLIPModel):
        """
        Evaluate the model on the image classification task.
        """
        classifier = HFCLIPClassifier(
            clip_model=clip_model, processor=self._clip_processor
        )
        classifier.set_classification_task(self.classnames, self.templates)
        if self._fabric is not None:
            classifier = self._fabric.setup_module(deepcopy(classifier))
        results = super().evaluate(classifier)
        log.info(f"Results for task {self.config.name}: {results}")
        return results


class CLIPImageClassificationTaskPool(TaskPool):
    _fabric: L.Fabric = None

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

    def prepare_task_config(self, task_config: DictConfig):
        # set default values for keys that are not present in per task configuration
        for key in ["num_workers", "batch_size", "fast_dev_run"]:
            if not hasattr(task_config, key):
                with open_dict(task_config):
                    task_config[key] = self.config[key]
        return task_config

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
        task_config = self.prepare_task_config(task_config)

        # load the task from the configuration
        task = CLIPImageClassificationTask(task_config)
        task._fabric = self._fabric
        task._taskpool = self
        task._clip_processor = self.clip_processor

        return task

    def evaluate(self, model: CLIPVisionModel):
        """
        Evaluate the model on the image classification task.
        """
        # if the fabric is not set, and we have a GPU, create a fabric instance
        if self._fabric is None and torch.cuda.is_available():
            self._fabric = L.Fabric(devices=1)
            self._fabric.launch()

        # CLIPVisionModel works the same with CLIPVisonTransformer, so we can use it directly
        self.clip_model.vision_model = model
        report = {}
        training_params, all_params = count_parameters(model)
        report["model_info"] = {
            "trainable_params": training_params,
            "all_params": all_params,
            "trainable_percentage": training_params / all_params,
        }
        for task_name in tqdm(self.task_names, desc="Evaluating tasks"):
            task = self.load_task(task_name)
            result = task.evaluate(self.clip_model)
            report[task_name] = result
        log.info(f"Results for taskpool {self.config.name}: {report}")
        if self._fabric.is_global_zero and self._fabric.logger is not None:
            with open(
                os.path.join(self._fabric.logger.log_dir, "report.json"), "w"
            ) as fp:
                json.dump(report, fp)
        return report
