import functools
import logging
import os
from copy import deepcopy

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers.models.clip.modeling_clip import CLIPEncoder

from fusion_bench.dataset import CLIPDataset
from fusion_bench.method.task_arithmetic.task_arithmetic import task_arithmetic_merge
from fusion_bench.mixins import CLIPClassificationMixin
from fusion_bench.modelpool import CLIPVisionModelPool
from fusion_bench.models.rankone_moe import RankOneMoE
from fusion_bench.utils.data import InfiniteDataLoader

from .rankone_moe import RankOneMoEAlgorithm

log = logging.getLogger(__name__)


class CLIPRankOneMoEAlgorithm(
    RankOneMoEAlgorithm,
    CLIPClassificationMixin,
):
    """
    CLIPRankOneMoEAlgorithm is a class that implements the RankOneMoEAlgorithm (https://github.com/EnnengYang/RankOne-MoE)
    for CLIP models. It extends the RankOneMoEAlgorithm and CLIPClassificationMixin classes.

    Attributes:
        modelpool (CLIPVisionModelPool): The model pool containing the CLIP models.
    """

    modelpool: CLIPVisionModelPool = None

    def load_checkpoint(self, model, checkpoint):
        """
        Load the checkpoint file.

        Args:
            model: The model to load the checkpoint into.
            checkpoint: The path to the checkpoint file.
        """
        state = {"model": model}
        self._fabric.load(checkpoint, state)

    def save_checkpoint(self, model, checkpoint):
        """
        Save the checkpoint file.

        Args:
            model: The model to save the checkpoint from.
            checkpoint: The path to the checkpoint file.
        """
        self._fabric.save(checkpoint, {"model": model})

    def construct_moe_model(self) -> RankOneMoE:
        """
        Construct the RankOne-MoE model using the models in the model pool.

        Returns:
            RankOne-MoE: The constructed MoE model.
        """
        base_model = self.modelpool.load_model("_pretrained_")
        expert_models = [
            self.modelpool.load_model(m) for m in self.modelpool.model_names
        ]

        # Merge the models using task arithmetic
        moe_model = task_arithmetic_merge(
            # This function modifies the model in place, so we need to pass a deepcopy
            deepcopy(base_model),
            expert_models,
            scaling_factor=self.config.init_lambda,
        ).requires_grad_(False)

        # Up-scale MLP modules
        base_encoder: CLIPEncoder = base_model.vision_model.encoder
        moe_encoder: CLIPEncoder = moe_model.vision_model.encoder
        expert_encoders = [m.vision_model.encoder for m in expert_models]

        num_layers = len(base_encoder.layers)
        for layer_idx in range(num_layers):
            base_mlp = base_encoder.layers[layer_idx].mlp
            expert_mlps = [e.layers[layer_idx].mlp for e in expert_encoders]

            moe_encoder.layers[layer_idx].mlp = RankOneMoE(
                hidden_size=base_encoder.config.hidden_size,
                base_model=base_mlp,
                expert_models=expert_mlps,
                init_lambda=self.config.init_lambda,
                batch_first=True,  # For open_clip models this is False
                router_hidden_layers=self.config.router_hidden_layers,
                batch_reduce=self.config.batch_reduce,
                svd_accelerator=self.config.svd_accelerator,
                rank_k=self.config.rank_k,
                select_k=self.config.select_k,
            )

        return moe_model

    @functools.cache
    def get_shuffled_test_loader_iter(self, tta_dataset: str):
        """
        Get an iterator for the shuffled test data loader.

        Args:
            tta_dataset (str): The name of the test-time adaptation dataset.

        Returns:
            Iterator: An iterator for the shuffled test data loader.
        """
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
        Load the CLIP processor and construct the zero-shot classification head for each task.
        """
        self.setup_zero_shot_classification_head()

    def compute_logits(self, module, batch, task) -> Tensor:
        """
        Compute the logits for the given batch and task.

        Args:
            module: The model module.
            batch: The input batch.
            task: The task name.

        Returns:
            Tensor: The computed logits.
        """
        images, _ = batch
        text_embeds = self.zeroshot_weights[task]

        image_embeds = module(images)[1]
        image_embeds = self.visual_projection(image_embeds)

        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        # Cosine similarity
        logits_per_text = (
            torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale_exp
        )
        logits_per_image = logits_per_text.t()

        return logits_per_image
