import logging
from abc import abstractmethod
from typing import List, Mapping, Union  # noqa: F401

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from fusion_bench.compat.method import ModelFusionAlgorithm
from fusion_bench.compat.modelpool import ModelPool
from fusion_bench.models.wrappers.task_wise_fusion import (
    TaskWiseMergedModel,
    get_task_wise_weights,
)

log = logging.getLogger(__name__)


def entropy_loss(logits: Tensor) -> Tensor:
    """
    Compute the entropy loss of a set of logits.

    Args:
        logits (Tensor): The logits to compute the entropy loss of.

    Returns:
        Tensor: The entropy loss of the logits.
    """
    probs = torch.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()


class TaskWiseAdaMergingAlgorithm(ModelFusionAlgorithm):
    _fabric: L.Fabric = None

    def __init__(self, algorithm_config: DictConfig):
        super().__init__(algorithm_config)

        if self._fabric is None and torch.cuda.is_available():
            self._fabric = L.Fabric(devices=self.config.get("devices", 1))
            self._fabric.launch()

    @torch.no_grad()
    def construct_task_wise_merged_model(self, modelpool: ModelPool):
        if self.config.weights is None:
            task_wise_weight = get_task_wise_weights(
                num_models=len(modelpool.model_names),
                init_values=self.config.init_values,
            )
        else:
            if isinstance(self.config.weights, str):
                # self.config.weights is a path to a .np or .pt file
                if self.config.weights.endswith(".pt"):
                    task_wise_weight = torch.load(
                        self.config.weights, map_location="cpu"
                    ).detach_()
                elif self.config.weights.endswith(".np"):
                    task_wise_weight = torch.from_numpy(
                        np.load(self.config.weights)
                    ).detach_()
                else:
                    raise ValueError(f"Unsupported file format: {self.config.weights}")
            else:
                try:
                    task_wise_weight = torch.tensor(
                        list(self.config.weights), dtype=torch.float32
                    )
                except ValueError:
                    raise ValueError(
                        f"Unsupported weights format: {self.config.weights}"
                    )

        pretrained_model = modelpool.load_model("_pretrained_")
        finetuned_models = [
            modelpool.load_model(name) for name in modelpool.model_names
        ]

        module = TaskWiseMergedModel(
            task_wise_weight=task_wise_weight,
            pretrained_model=pretrained_model,
            finetuned_models=finetuned_models,
            clamp_weights=self.config.clamp_weights,
            tie_weights=self.config.tie_weights,
            strict=self.config.strict,
        )
        return module

    def run(self, modelpool: ModelPool):
        log.info("Fusing models using task-wise adaptive merging.")
        self.modelpool = modelpool

        module = self.construct_task_wise_merged_model(modelpool)

        if self.config.weights is not None:
            # skip the test-time adaptation
            return module.merge_and_unload()
        else:
            module = self.test_time_adaptation(module)
            if self.config.get("save_merging_weights", False):
                torch.save(module.merge_weight, self.config.save_merging_weights)
            return module.merge_and_unload()

    def on_test_time_adaptation_start(self):
        pass

    @abstractmethod
    def get_shuffled_test_loader_iter(self, task: str) -> DataLoader:
        pass

    @abstractmethod
    def compute_logits(self, module: nn.Module, batch, task: str) -> Tensor:
        """
        Compute the logits for the given batch and task.

        Args:
            module (nn.Module): The model module.
            batch (tuple): A batch of input data.
            task (str): The name of the task.

        Returns:
            Tensor: The classification logits for the batch.
        """
        pass

    def test_time_adaptation(self, module: TaskWiseMergedModel):
        self.on_test_time_adaptation_start()

        # configure optimizer
        if self.config.optimizer == "adam":
            optimizer = torch.optim.Adam([module.merge_weight], lr=self.config.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        if self._fabric is not None:
            module, optimizer = self._fabric.setup(module, optimizer)

        module.train()
        module.merge_weights()

        if self.config.get("fast_dev_run", False):
            log.info("Running fast_dev_run, only one step")
            pbar = tqdm(
                range(1),
                "AdaMerging Test-time adaptation",
                dynamic_ncols=True,
            )
        else:
            pbar = tqdm(
                range(self.config.max_steps),
                "AdaMerging Test-time adaptation",
                dynamic_ncols=True,
            )
        for step_idx in pbar:
            for task in self.modelpool.model_names:
                batch = next(self.get_shuffled_test_loader_iter(task))
                logits = self.compute_logits(module, batch, task)
                assert (
                    logits.dim() == 2
                ), f"Expected logits to be 2D, got {logits.dim()}"
                loss = entropy_loss(logits)
                # .backward() accumulates when .zero_grad() wasn't called
                # this can save memory
                self._fabric.backward(loss, retain_graph=True)

            optimizer.step()
            optimizer.zero_grad()
            module.merge_weights()

        return module
