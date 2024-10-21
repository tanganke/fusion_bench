import functools
import logging
import os
from copy import deepcopy

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPEncoder

from fusion_bench.dataset import CLIPDataset
from fusion_bench.method.task_arithmetic.task_arithmetic import task_arithmetic_merge
from fusion_bench.mixins import CLIPClassificationMixin
from fusion_bench.modelpool import CLIPVisionModelPool
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.models.we_moe import WeightEnsemblingMoE
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.utils import timeit_context
from fusion_bench.utils.data import InfiniteDataLoader

from .we_moe import WeightEnsemblingMoEAlgorithm

log = logging.getLogger(__name__)


class CLIPWeightEnsemblingMoEAlgorithm(
    WeightEnsemblingMoEAlgorithm,
    CLIPClassificationMixin,
):
    modelpool: CLIPVisionModelPool = None

    def load_checkpoint(self, model, checkpoint):
        state = {"model": model}
        self._fabric.load(checkpoint, state)

    def save_checkpoint(self, model, checkpoint):
        self._fabric.save(checkpoint, {"model": model})

    def construct_moe_model(self) -> WeightEnsemblingMoE:
        base_model = self.modelpool.load_model("_pretrained_")
        expert_models = [
            self.modelpool.load_model(m) for m in self.modelpool.model_names
        ]

        # merge the models using task arithmetic
        moe_model = task_arithmetic_merge(
            # this function modifies the model in place, so we need to pass a deepcopy
            deepcopy(base_model),
            expert_models,
            scaling_factor=self.config.init_lambda,
        ).requires_grad_(False)

        # up-scale MLP modules
        base_encoder: CLIPEncoder = base_model.vision_model.encoder
        moe_encoder: CLIPEncoder = moe_model.vision_model.encoder
        expert_encoders = [m.vision_model.encoder for m in expert_models]

        num_layers = len(base_encoder.layers)
        for layer_idx in range(num_layers):
            base_mlp = base_encoder.layers[layer_idx].mlp
            expert_mlps = [e.layers[layer_idx].mlp for e in expert_encoders]

            moe_encoder.layers[layer_idx].mlp = WeightEnsemblingMoE(
                hidden_size=base_encoder.config.hidden_size,
                base_model=base_mlp,
                expert_models=expert_mlps,
                init_lambda=self.config.init_lambda,
                batch_first=True,  # for open_clip models this is False
                router_hidden_layers=self.config.router_hidden_layers,
                batch_reduce=self.config.batch_reduce,
            )

        return moe_model

    @functools.cache
    def get_shuffled_test_loader_iter(self, tta_dataset: str):
        dataset = self.modelpool.load_test_dataset(tta_dataset)
        dataset = CLIPDataset(dataset, processor=self.clip_processor)
        log.info("get_shuffled_test_loader_iter")
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        loader = self.fabric.setup_dataloaders(loader)
        return iter(InfiniteDataLoader(loader))

    def on_test_time_adaptation_start(self):
        """
        Here we load the CLIP processor and construct the zero-shot classification head for each task.
        """
        self.setup_zero_shot_classification_head()

    def compute_logits(self, module, batch, task) -> Tensor:
        images, _ = batch
        text_embeds = self.zeroshot_weights[task]

        image_embeds = module(images)[1]
        image_embeds = self.visual_projection(image_embeds)

        # normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity
        logits_per_text = (
            torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale_exp
        )
        logits_per_image = logits_per_text.t()

        return logits_per_image
