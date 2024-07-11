"""
Examples:

fusion_bench \
    fabric_logger.name=ViT-B-32/pge_task_arithmetic \
    method=clip_pge_task_arithmetic \
    modelpool=clip-vit-base-patch32_TA8 \
    taskpool=clip-vit-classification_TA8
"""

import functools
import itertools
import logging
import os
from copy import deepcopy
from functools import cache
from typing import Dict, List, Tuple, cast

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import CLIPVisionModel

from fusion_bench.method import ModelFusionAlgorithm
from fusion_bench.method.adamerging.entropy_loss import entropy_loss
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import ModelPool, to_modelpool
from fusion_bench.modelpool.huggingface_clip_vision import HuggingFaceClipVisionPool
from fusion_bench.models.masks import MaskModel, PGEMaskModel, mask_sparsity
from fusion_bench.models.wrappers.task_wise_fusion import (
    TaskWiseMergedModel,
    get_task_wise_weights,
)
from fusion_bench.tasks.clip_classification.clip_mixin import CLIPClassificationMixin
from fusion_bench.utils.data import InfiniteDataLoader
from fusion_bench.utils.parameters import print_parameters
from fusion_bench.utils.type import _StateDict

log = logging.getLogger(__name__)


class PGETaskArithmeticAlgorithmForCLIP(
    CLIPClassificationMixin,
    SimpleProfilerMixin,
    ModelFusionAlgorithm,
):
    @torch.no_grad()
    def setup_models(self):
        modelpool = self.modelpool

        # Load the pretrained model
        pretrained_model = modelpool.load_model("_pretrained_")

        # construct PGE mask model
        mask_model = PGEMaskModel(
            pretrained_model,
            ignore_untrained_params=True,
        )
        mask_model.fill_(self.config.initial_p)
        print("Summary of mask model:")
        print_parameters(mask_model)

        # Load the fine-tuned models
        finetuned_models = [
            modelpool.load_model(name) for name in modelpool.model_names
        ]

        task_wise_weight = get_task_wise_weights(
            num_models=len(modelpool.model_names),
            init_values=self.config.scaling_factor,
        )

        module = TaskWiseMergedModel(
            task_wise_weight=task_wise_weight,
            pretrained_model=pretrained_model,
            finetuned_models=finetuned_models,
            clamp_weights=self.config.clamp_weights,
            tie_weights=self.config.tie_weights,
            strict=self.config.strict,
        )
        return module, mask_model

    def train_mask(self, module: TaskWiseMergedModel, mask_model: PGEMaskModel):
        # configure optimizer
        lr_scheduler = None
        if self.config.optimizer == "adam":
            optimizer = torch.optim.Adam(mask_model.parameters(), lr=self.config.lr)
            # print(f"{optimizer=}")
            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer, self.config.max_steps, eta_min=0.1
            # )
        elif self.config.optimizer == "sgd":
            optimizer = torch.optim.SGD(mask_model.parameters(), lr=self.config.lr)
            print(f"{optimizer=}")
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.config.max_steps, eta_min=0.1
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        mask_model.train()
        for step_idx in (
            pbar := tqdm(
                range(self.config.max_steps if not self.is_debug_mode else 1),
                ("[DEBUG MODE] " if self.is_debug_mode else "")
                + "PGE Task Arithmetic Test-time adaptation",
                dynamic_ncols=True,
            )
        ):
            metrics = {}
            with torch.no_grad():  # no back-probagation is needed
                # sample a batch of images for each task
                with self.profile("sample mask"):
                    mask = mask_model.sample_mask(mask_type="discrete")
                    metrics["train/sparisity"] = mask_sparsity(mask)
                with self.profile("merge weights"):
                    module.merge_weights(task_vector_mask=mask)
                loss = 0
                for task in self.modelpool.model_names:
                    with self.profile("data loading"):
                        batch = next(self.get_shuffled_test_loader_iter(task))
                        images = batch[0]
                    with self.profile("forward pass"):
                        logits = self.compute_logits(module, images, task)
                        loss += entropy_loss(logits)
                with self.profile("compute grad"):
                    mask_model.compute_grad(loss, mask)

            with self.profile("optimizer step"):
                optimizer.step()
                optimizer.zero_grad()
                mask_model.clamp_(0.01, 0.99)

                if lr_scheduler is not None:
                    lr_scheduler.step()

            metrics.update({"train/loss": loss.item()})
            self.fabric.log_dict(metrics, step=step_idx)
            pbar.set_postfix(metrics)

            if (step_idx + 1) % self.config.save_interval == 0:
                with self.profiler.profile("save checkpoint"):
                    save_dir = os.path.join(self.fabric.logger.log_dir, "checkpoints")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"pge_mask_steps_{step_idx}.pt")
                    print(f"saving checkpoint to {save_path}")
                    state = {"model": mask_model}
                    self.fabric.save(save_path, state)

                    # Create or update a symbolic link to the latest checkpoint
                    if self.fabric.is_global_zero:
                        symlink_path = os.path.join(save_dir, "latest_checkpoint.pt")
                        if os.path.exists(symlink_path):
                            os.remove(symlink_path)
                        os.link(os.path.abspath(save_path), symlink_path)

                self.print_profile_summary()

        return module

    def run(self, modelpool: HuggingFaceClipVisionPool):
        self.modelpool = to_modelpool(modelpool)
        config = self.config
        self.log_hyperparams(config, filename="method_config.yaml")

        with self.profile("setup models"):
            module, mask_model = self.setup_models()
            mask_model: PGEMaskModel = self.fabric.to_device(mask_model)
            module: TaskWiseMergedModel = self.fabric.to_device(module)

        if config.mask_checkpoint is None:
            self.setup_zero_shot_classification_head()
            self.train_mask(module, mask_model)
        else:
            if self.fabric.is_global_zero:
                print("loading mask from checkpoint", config.mask_checkpoint)
            self.fabric.load(config.mask_checkpoint, {"model": mask_model})

        mask = mask_model.sample_mask(mask_type="discrete")
        return module.merge_and_unload(mask)
