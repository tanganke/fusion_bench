import logging
import os
from abc import abstractmethod
from copy import deepcopy
from typing import Any, List, Mapping, Union, cast

import lightning as L
import numpy as np
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from fusion_bench.method.base_algorithm import ModelFusionAlgorithm
from fusion_bench.mixins.lightning_fabric import LightningFabricMixin
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import ModelPool
from fusion_bench.models.wrappers.layer_wise_fusion import (
    LayerWiseMergedModel,
    get_layer_wise_weights,
)
from fusion_bench.utils.data import load_tensor_from_file
from fusion_bench.utils.state_dict_arithmetic import state_dict_sub
from fusion_bench.utils.type import _StateDict

from .entropy_loss import entropy_loss

log = logging.getLogger(__name__)


class LayerWiseAdaMergingAlgorithm(
    ModelFusionAlgorithm,
    LightningFabricMixin,
    SimpleProfilerMixin,
):
    def __init__(self, algorithm_config: DictConfig):
        super().__init__(algorithm_config)

    @torch.no_grad()
    def construct_layer_wise_merged_model(self, modelpool: ModelPool):
        """
        Constructs a wrapped layer-wise merged model from model pool.

        This method creates a new wrapped model by merging the layers of a pretrained model with those of several fine-tuned models.
        The merging is controlled by layer-wise weights, which is a `torch.Tensor` of the shape `(num_models, num_layers)`.
        The merging weights can be initialized based on a provided configuration or loaded from a file.

        Args:
            modelpool (ModelPool): An object containing the pretrained model and fine-tuned models to be merged.

        Returns:
            LayerWiseMergedModel: An instance of the merged model with layer-wise weights applied.
        """
        pretrained_model = modelpool.load_model("_pretrained_")
        finetuned_models = [
            modelpool.load_model(name) for name in modelpool.model_names
        ]

        # initialize layer-wise weights using the provided configuration `init_values` or load from file if `weights` is provided
        if self.config.weights is None:
            layer_wise_weight = get_layer_wise_weights(
                num_models=len(modelpool.model_names),
                num_layers=len(
                    tuple(
                        filter(lambda p: p.requires_grad, pretrained_model.parameters())
                    )
                ),
                init_values=self.config.init_values,
            )
        else:
            if isinstance(self.config.weights, str):
                # self.config.weights is a path to a saved tensor
                layer_wise_weight = load_tensor_from_file(self.config.weights)
            else:
                raise ValueError(f"Unsupported weights format: {self.config.weights}")

        module = LayerWiseMergedModel(
            layer_wise_weight=layer_wise_weight,
            pretrained_model=pretrained_model,
            finetuned_models=finetuned_models,
            clamp_weights=self.config.clamp_weights,
            tie_weights=self.config.tie_weights,
            strict=self.config.strict,
        )
        print(f"{layer_wise_weight.size()=}, {layer_wise_weight.numel()=}")
        return module

    @rank_zero_only
    def save_merging_weights(self, file_path: str, merging_weights: torch.Tensor):
        if self.fabric.is_global_zero and self.config.get(
            "save_merging_weights", False
        ):
            if isinstance(file_path, str) and not file_path.startswith(("/", ".")):
                # if the file path is not absolute or relative to current working directory, save it in the log directory
                save_path = os.path.join(self.log_dir, file_path)
            else:
                save_path = file_path
            log.info(f"saving merging weights to {save_path}.")
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(merging_weights.detach().cpu(), save_path)

    def run(self, modelpool: ModelPool):
        log.info("Fusing models using layer-wise adaptive merging.")
        self.modelpool = modelpool
        self.log_hyperparams(self.config)

        with self.profile("construct the wrapped model"):
            module = self.construct_layer_wise_merged_model(modelpool)

        if self.config.weights is not None:
            # skip the test-time adaptation
            return module.merge_and_unload()
        else:
            with self.profile("test-time adaptation"):
                module = self.test_time_adaptation(module)
            if self.config.get("save_merging_weights", False):
                self.save_merging_weights(
                    self.config.save_merging_weights, module.merge_weight
                )
            return module.merge_and_unload()

    def on_test_time_adaptation_start(self):
        """
        Something to do before the test-time adaptation starts. Such as setting up the task-specific heads.
        """
        pass

    @abstractmethod
    def get_shuffled_test_loader_iter(self, task: str) -> DataLoader:
        """
        Loader of test dataset for test-time adaptation. labels are not needed.
        """
        pass

    @abstractmethod
    def compute_logits(self, module, images: Tensor, task: str) -> Tensor:
        pass

    def test_time_adaptation(self, module: LayerWiseMergedModel):
        self.on_test_time_adaptation_start()
        config = self.config

        # configure optimizer
        if self.config.optimizer == "adam":
            optimizer = torch.optim.Adam([module.merge_weight], lr=self.config.lr)
            print(f"{optimizer=}")
            module, optimizer = self.fabric.setup(module, optimizer)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        module.train()
        module.merge_weights()
        for step_idx in (
            pbar := tqdm(
                range(self.config.max_steps if not self.is_debug_mode else 1),
                ("[DEBUG MODE] " if self.is_debug_mode else "")
                + "AdaMerging Test-time adaptation",
                dynamic_ncols=True,
            )
        ):
            # default behavior for first-order optimizers
            for task in self.modelpool.model_names:
                with self.profile("data loading"):
                    batch = next(self.get_shuffled_test_loader_iter(task))
                with self.profile("forward pass"):
                    logits = self.compute_logits(module, batch[0], task)
                    loss = entropy_loss(logits)
                with self.profile("backward pass"):
                    self.fabric.backward(loss, retain_graph=True)

            with self.profile("optimizer step"):
                optimizer.step()
                optimizer.zero_grad()
            with self.profile("merging weights"):
                module.merge_weights()

            metrics = {
                "train/loss": loss.item(),
                "train/weight_max": module.merge_weight.max().item(),
                "train/weight_min": module.merge_weight.min().item(),
                "train/weight_mean": module.merge_weight.mean().item(),
            }
            self.fabric.log_dict(metrics, step=step_idx)
            pbar.set_postfix(metrics)

        self.print_profile_summary()
        return module
