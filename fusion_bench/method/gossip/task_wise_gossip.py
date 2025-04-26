import copy
import gc
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


# obtain the current GPU memory usage
def print_memory_usage(desc):
    print(desc)
    allocated = torch.cuda.memory_allocated() / 1024**2  # 转换为 MB
    cached = torch.cuda.memory_reserved() / 1024**2  # 转换为 MB
    print(f"Allocated Memory: {allocated:.2f} MB")
    print(f"Cached Memory: {cached:.2f} MB")


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


class ModelScheduler:
    """
    Manage the storage of models, schedule the order in which models are loaded to GPU
    transfer data between the CPU and GPU
    """

    def __init__(
        self,
        modelpool: ModelPool,
        config: DictConfig,
    ):
        self.pretrained_model = modelpool.load_model("_pretrained_")
        self.finetuned_models = [
            modelpool.load_model(name) for name in modelpool.model_names
        ]
        self.num_finetuned_models = len(self.finetuned_models)
        self.new_finetuned_models = copy.deepcopy(self.finetuned_models)
        self.finetuned_model_names = [name for name in modelpool.model_names]

        self.config = config

    @torch.no_grad()  # not sure whether to use this
    def __call__(self, model_id):
        """
        return models and relevant data in each step
        """
        # TODO: use a mixing matrix to determine which models to use in step idx

        pretrained_model = copy.deepcopy(self.finetuned_models[model_id])
        finetuned_models = [
            copy.deepcopy(
                self.finetuned_models[(model_id + 1) % self.num_finetuned_models]
            ),
            copy.deepcopy(
                self.finetuned_models[(model_id - 1) % self.num_finetuned_models]
            ),
        ]

        if self.config.weights is None:
            task_wise_weight = get_task_wise_weights(
                num_models=len(finetuned_models),
                init_values=self.config.init_values,
            )
        else:
            pass

        module = TaskWiseMergedModel(
            task_wise_weight=task_wise_weight,
            pretrained_model=pretrained_model,
            finetuned_models=finetuned_models,
            clamp_weights=self.config.clamp_weights,
            tie_weights=self.config.tie_weights,
            strict=self.config.strict,
        )
        return module

    def store_model(self, new_finetuned_model_dict, model_id):
        """
        store new finetuned model after every turn of adamerging
        """
        self.new_finetuned_models[model_id].load_state_dict(new_finetuned_model_dict)

    def update_models(self):
        self.finetuned_models = copy.deepcopy(self.new_finetuned_models)

    def get_final_models(self):
        # need a check
        final_models = [
            {"name": name, "model": model}
            for name, model in zip(self.finetuned_model_names, self.finetuned_models)
        ]
        num_finetuned_models = len(self.finetuned_models)

        state_dict = self.pretrained_model.state_dict(keep_vars=True)
        for name in state_dict.keys():
            state_dict[name].data.zero_()
        for model in self.finetuned_models:
            for name, param in model.named_parameters():
                state_dict[name] = state_dict[name] + 1 / num_finetuned_models * param

        self.pretrained_model.load_state_dict(state_dict)
        final_models += [{"name": "average model", "model": self.pretrained_model}]

        return final_models


class TaskWiseGossipAlgorithm(ModelFusionAlgorithm):
    _fabric: L.Fabric = None

    def __init__(self, algorithm_config: DictConfig):
        super().__init__(algorithm_config)

        if self._fabric is None and torch.cuda.is_available():
            self._fabric = L.Fabric(devices=self.config.get("devices", 1))
            self._fabric.launch()

        self.optimizer = None  # we want to reuse it in Gossip using single GPU

    def free_gpu_memory(self, module: TaskWiseMergedModel):
        module.pretrained_model.to("cpu")
        for model in module.task_vectors:
            model.to("cpu")
        del module
        gc.collect()
        torch.cuda.empty_cache()
        print_memory_usage(
            "finish local adamerging, after freeing memory, the memory usage of GPU is:"
        )

    def run(self, modelpool: ModelPool):
        log.info("Fusing models using task-wise adaptive merging with gossip.")
        self.modelpool = modelpool
        self.num_finetuned_models = len(modelpool.model_names)

        model_scheduler = ModelScheduler(self.modelpool, self.config)

        pbar = tqdm(
            range(self.config.gossip_max_steps), "Gossip merging", dynamic_ncols=True
        )
        for step_idx in pbar:
            log.info(f"step: {step_idx}")
            for model_id in tqdm(
                range(self.num_finetuned_models), "local adamerging", dynamic_ncols=True
            ):
                # log.info(f"adamerging model: {model_scheduler.finetuned_midels_name[model_id]}")
                module = model_scheduler(model_id)
                module = self.test_time_adaptation(module)
                # if self.config.get("save_merging_weights", False):
                #     torch.save(module.merge_weight, self.config.save_merging_weights)
                print_memory_usage(
                    "local adamerging almost done, the memory usage of GPU is:"
                )
                model_scheduler.store_model(module.merge_weights(), model_id)
                print_memory_usage(
                    "local adamerging almost done, the memory usage of GPU is:"
                )
                self.free_gpu_memory(
                    module
                )  # simulate distributed GPU memory usage as much as possible

            model_scheduler.update_models()

        return model_scheduler.get_final_models()

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
            self.optimizer = torch.optim.Adam([module.merge_weight], lr=self.config.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        if self._fabric is not None:
            module, self.optimizer = self._fabric.setup(module, self.optimizer)
        print_memory_usage(
            "load model and optimizer to GPU, the memory usage of GPU is:"
        )
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

            # print_memory_usage('model + dataset: ')
            self.optimizer.step()
            self.optimizer.zero_grad()
            module.merge_weights()

        del self.optimizer
        gc.collect()
        torch.cuda.empty_cache()
        return module
