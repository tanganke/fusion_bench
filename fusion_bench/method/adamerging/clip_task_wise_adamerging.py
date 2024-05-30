import functools
import itertools
import logging
import os

import torch
from omegaconf import DictConfig, open_dict
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from fusion_bench.dataset import CLIPDataset, load_dataset_from_config
from fusion_bench.modelpool.huggingface_clip_vision import HuggingFaceClipVisionPool
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.utils import timeit_context

from .task_wise_adamerging import TaskWiseAdaMergingAlgorithm

log = logging.getLogger(__name__)


class InfiniteDataLoader:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)  # Reset the data loader
            data = next(self.data_iter)
        return data


class CLIPTaskWiseAdaMergingAlgorithm(TaskWiseAdaMergingAlgorithm):
    modelpool: HuggingFaceClipVisionPool = None
    _clip_processor: CLIPProcessor = None
    zeroshot_weights = {}

    def __init__(self, algorithm_config: DictConfig):
        super().__init__(algorithm_config)

    def get_task_config(self, task):
        for task_config in self.modelpool.config.tta_datasets:
            if task_config.name == task:
                return task_config
        raise ValueError(f"Task {task} not found in config")

    def prepare_dataset_config(self, dataset_config: DictConfig):
        if not hasattr(dataset_config, "type"):
            with open_dict(dataset_config):
                dataset_config["type"] = self.modelpool.config.dataset_type
        return dataset_config

    @functools.cache
    def get_test_dataset(self, task: str):
        """
        Load the test dataset for the task.
        This method is cached, so the dataset is loaded only once.
        """
        dataset_config = self.get_task_config(task)["dataset"]
        dataset_config = self.prepare_dataset_config(dataset_config)
        log.info(f"Loading test dataset: {dataset_config.name}")
        dataset = load_dataset_from_config(dataset_config)
        dataset = CLIPDataset(dataset, self._clip_processor)
        return dataset

    @functools.cache
    def get_shuffled_test_loader_iter(self, task: str):
        loader = DataLoader(
            self.get_test_dataset(task),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        if self._fabric is not None:
            loader = self._fabric.setup_dataloaders(loader)
        return iter(InfiniteDataLoader(loader))

    def on_test_time_adaptation_start(self):
        """
        Here we load the CLIP processor and construct the zero-shot classification head for each task.
        """
        clip_model_config = self.modelpool.get_model_config("_pretrained_")

        with timeit_context("Loading CLIP processor and pretrained CLIP model."):
            self._clip_processor = CLIPProcessor.from_pretrained(clip_model_config.path)
            clip_model = CLIPModel.from_pretrained(clip_model_config.path)

            clip_classifier = HFCLIPClassifier(clip_model, self._clip_processor)
            self.visual_projection = clip_model.visual_projection.requires_grad_(False)
            self.logit_scale = clip_model.logit_scale.exp()
            if self._fabric is not None:
                self.visual_projection = self._fabric.to_device(self.visual_projection)
                self.logit_scale = self._fabric.to_device(self.logit_scale)

        for task in self.modelpool.model_names:
            cache_file = os.path.join(
                self.config.cache_dir,
                f"{os.path.basename(clip_model_config.path)}_{task}_zeroshot_weights.pt",
            )
            if os.path.exists(cache_file):
                log.info(f"Loading cached zeroshot weights for task: {task}")
                zeroshot_weights = torch.load(cache_file, map_location="cpu")
            else:
                log.info(f"Construct zero shot classification head for task: {task}")
                classnames, templates = get_classnames_and_templates(
                    self.get_task_config(task)["dataset"].name
                )
                clip_classifier.set_classification_task(classnames, templates)
                zeroshot_weights = clip_classifier.zeroshot_weights
                log.info(f"save zeroshot weights to {cache_file}")
                torch.save(zeroshot_weights, cache_file)
            self.zeroshot_weights[task] = zeroshot_weights
            if self._fabric is not None:
                self.zeroshot_weights[task] = self._fabric.to_device(
                    self.zeroshot_weights[task]
                )

    def compute_logits(self, module, batch, task: str) -> Tensor:
        images, _ = batch
        text_embeds = self.zeroshot_weights[task]

        image_embeds = module(images)[1]
        image_embeds = self.visual_projection(image_embeds)

        # normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale
        logits_per_image = logits_per_text.t()

        return logits_per_image
