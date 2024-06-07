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

from fusion_bench.modelpool.huggingface_clip_vision import HuggingFaceClipVisionPool
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.utils import timeit_context

from .regmean import RegMeanAlgorithm

log = logging.getLogger(__name__)


class RegMeanAlgorithmForCLIP(RegMeanAlgorithm):
    _fabric: L.Fabric = None
    _clip_processor: CLIPProcessor = None
    zeroshot_weights = {}

    def __init__(self, algorithm_config: DictConfig):
        super().__init__(algorithm_config)

        # setup fabric
        if self._fabric is None and torch.cuda.is_available():
            self._fabric = L.Fabric(devices=self.config.devices)
            self._fabric.launch()

    def on_regmean_start(self):
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

    def get_regmean_weights(
        self,
        model_name: str,
        model: Module,
        train_dataset,
        linear_modules_to_merge: Dict[str, Module],
    ):
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

        def compute_regmean_weights(module_name: str):
            """
            compute the regmean weights, a hook function to deal with each module's input
            :param module_name: str, module name
            :return:
            """

            def hook(module: nn.Module, input: tuple, output: torch.Tensor):
                # Tensor, shape (batch_size, sequence_length, hidden_dim)
                x = cast(Tensor, input[0]).detach()
                batch_num_actual_examples = x.shape[0]
                # Tensor, shape (batch_size * sequence_length, hidden_dim)
                x = x.reshape(-1, x.shape[-1])
                # Tensor, shape (hidden_dim, hidden_dim)
                xtx = torch.matmul(x.transpose(0, 1), x)
                # store the averaged weights in regmean_weights
                if module_name not in regmean_weights.keys():
                    regmean_weights[module_name] = xtx / x.shape[0]
                    num_computed_examples[module_name] = x.shape[0]
                    num_actual_examples[module_name] = batch_num_actual_examples
                else:
                    regmean_weights[module_name] = (
                        regmean_weights[module_name]
                        * num_computed_examples[module_name]
                        + xtx
                    ) / (num_computed_examples[module_name] + x.shape[0])
                    num_computed_examples[module_name] += x.shape[0]
                    num_actual_examples[module_name] += batch_num_actual_examples

            return hook

        handles = []
        # dictionary, regmean matrices for each linear module inputs
        regmean_weights = {}
        # dictionary, number of examples (multiplied the sequence length) used for computing regmean matrices
        num_computed_examples = {}
        # dictionary, number of actual examples used for computing regmean matrices
        num_actual_examples = {}

        for module_name, linear_module_to_merge in linear_modules_to_merge.items():
            # register a hook in the forward process
            handle = linear_module_to_merge.register_forward_hook(
                compute_regmean_weights(module_name=module_name)
            )
            handles.append(handle)
        for step, batch in tqdm(
            enumerate(train_dataloader),
            desc=f"computing regmean weights for model {model_name}",
        ):
            if (
                len(num_actual_examples) > 0
                and list(num_actual_examples.values())[0]
                >= self.config.num_regmean_examples
            ):
                break
            logits = self.compute_logits(model, batch, model_name)

        # remove the added hook
        for handle in handles:
            handle.remove()

        for module_name in regmean_weights.keys():
            regmean_weights[module_name] = regmean_weights[module_name].detach().cpu()

        return regmean_weights
