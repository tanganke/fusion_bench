import logging
import os
from copy import deepcopy
from functools import cache
from typing import Dict, List, cast

import lightning as L
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn.modules import Module
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel

from fusion_bench.dataset import CLIPDataset, load_dataset_from_config
from fusion_bench.modelpool.huggingface_clip_vision import HuggingFaceClipVisionPool
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.utils import timeit_context

from .fisher_merging import FisherMergingAlgorithm, get_param_squared_gradients

log = logging.getLogger(__name__)


class FisherMergingAlgorithmForCLIP(FisherMergingAlgorithm):
    _fabric: L.Fabric = None
    _clip_processor: CLIPProcessor = None
    zeroshot_weights = {}

    def __init__(self, algorithm_config: DictConfig):
        super().__init__(algorithm_config)

        # setup fabric
        if self._fabric is None and torch.cuda.is_available():
            self._fabric = L.Fabric(devices=self.config.devices)
            self._fabric.launch()

    def on_fisher_merging_start(self):
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
                    cast(HuggingFaceClipVisionPool, self.modelpool)
                    .get_train_dataset_config(task)["dataset"]
                    .name
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

    def get_fisher_weights(
        self,
        model_name: str,
        model: Module,
        train_dataset,
        param_names_to_merge: List[str],
    ) -> Dict[str, Tensor]:
        # setup dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        if self._fabric is not None:
            train_dataloader = self._fabric.setup_dataloaders(train_dataloader)
            model = self._fabric.setup(model)
        num_fisher_examples = self.config.num_fisher_examples
        if num_fisher_examples % train_dataloader.batch_size != 0:
            print(
                f"warning: the number of examples for computing fisher cannot be fully divided by the batch size for model, "
                "which may lead to a slightly different number of the actually used examples."
            )
        num_computed_examples = 0
        batches_fisher_weights_list = []
        for step, batch in tqdm(
            enumerate(train_dataloader),
            desc=f"computing fisher weights",
            total=num_fisher_examples // train_dataloader.batch_size,
        ):
            if num_computed_examples >= num_fisher_examples:
                break
            logits = self.compute_logits(model, batch, model_name)
            # Tensor, shape (batch_size, num_label_classes)

            # compute fisher weights for classifxication task
            # use detach() to detach from the computation graph
            # Tensor, shape (batch_size, num_label_classes)
            labels_probabilities = torch.softmax(logits, dim=-1).detach()
            labels_log_probabilities = torch.log_softmax(logits, dim=-1)
            # sqrt labels_probabilities, since torch.sqrt(labels_probabilities) would be squared in the following squared gradients
            labels_expectations = (
                torch.sqrt(labels_probabilities) * labels_log_probabilities
            )
            # sum over label classes and batch dimension
            sum_labels_expectations = labels_expectations.sum(dim=-1).sum(dim=0)
            model.zero_grad()
            sum_labels_expectations.backward()
            # dict, fisher weights of a batch
            batch_fisher_weights = get_param_squared_gradients(
                model=model, param_names_to_merge=param_names_to_merge
            )

            # move fisher weights to cpu to save GPU memory
            for key, weights in batch_fisher_weights.items():
                batch_fisher_weights[key] = weights.detach().cpu()

            batches_fisher_weights_list.append(batch_fisher_weights)
            num_computed_examples += batch[0].size(0)

        model_to_merge_fisher_weights = {}
        for batch_fisher_weights in batches_fisher_weights_list:
            for key in batch_fisher_weights:
                if key not in model_to_merge_fisher_weights:
                    model_to_merge_fisher_weights[key] = batch_fisher_weights[key]
                else:
                    model_to_merge_fisher_weights[key] += batch_fisher_weights[key]

        # mean over batches
        for key in model_to_merge_fisher_weights:
            model_to_merge_fisher_weights[key] /= num_computed_examples
            model_to_merge_fisher_weights[key] = (
                model_to_merge_fisher_weights[key].detach().cpu()
            )
        return model_to_merge_fisher_weights
