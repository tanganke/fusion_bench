"""
This is an experimental implementation of the Layer-Wise AdaMerging Algorithm for Flan-T5 models.
The efficiency of the algorithm is not guaranteed, and it may not work as expected.
"""

import functools
import gc
import logging
import os
from abc import abstractmethod
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Union, cast  # noqa: F401

import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig, ListConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import T5ForConditionalGeneration
from transformers.data import default_data_collator

from fusion_bench.compat.modelpool.base_pool import ModelPool
from fusion_bench.method import BaseAlgorithm
from fusion_bench.method.simple_average import simple_average
from fusion_bench.mixins.lightning_fabric import LightningFabricMixin
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import Seq2SeqLMPool
from fusion_bench.models.wrappers.layer_wise_fusion import (
    LayerWiseMergedModel,
    get_layer_wise_weights,
)
from fusion_bench.utils.data import InfiniteDataLoader, load_tensor_from_file
from fusion_bench.utils.instantiate import instantiate

from .entropy_loss import entropy_loss
from .layer_wise_gossip import ModelScheduler
from .min_norm_solvers import MinNormSolver
from .utils import get_memory_usage

log = logging.getLogger(__name__)


class FlanT5LayerWiseGossipAlgorithm(
    BaseAlgorithm,
    LightningFabricMixin,
    SimpleProfilerMixin,
):

    def __init__(
        self,
        optimizer: DictConfig,
        dataloader_kwargs: DictConfig,
        init_values: float,
        max_steps: int,
        merging_weights_load_path: Optional[Union[str, Path]] = None,
        merging_weights_save_path: Optional[Union[str, Path]] = None,
        clamp_weights: bool = False,
        tie_weights: bool = True,
        strict: bool = False,
        cache_dir: str = "outputs/cache",
        variant: Optional[str] = None,
        **kwargs,
    ):
        self._optimizer = optimizer
        self.dataloader_kwargs = dataloader_kwargs
        self.init_values = init_values
        self.merging_weights_load_path = merging_weights_load_path
        self.merging_weights_save_path = merging_weights_save_path
        self.clamp_weights = clamp_weights
        self.tie_weights = tie_weights
        self.strict = strict
        self.max_steps = max_steps
        self.cache_dir = cache_dir
        self.variant = variant

        self.configs = SimpleNamespace(**kwargs)
        self.configs.init_values = init_values
        self.configs.clamp_weights = clamp_weights
        self.configs.tie_weights = tie_weights
        self.configs.strict = strict
        if isinstance(self.configs.accuracy_test_interval, ListConfig):
            self.configs.accuracy_test_interval = list(
                self.configs.accuracy_test_interval
            )
        elif isinstance(self.configs.accuracy_test_interval, int):
            pass
        else:
            log.warning(
                f"Unexpected type of accuracy_test_interval: {type(self.configs.accuracy_test_interval)}"
            )
        super().__init__(**kwargs)

    @rank_zero_only
    def save_merging_weights(self, file_path: str, merging_weights: torch.Tensor):
        """
        Save the merging weights to a file.

        Args:
            file_path (str): The path to save the merging weights.
            merging_weights (torch.Tensor): The merging weights to save.
        """
        if self.fabric.is_global_zero and self.merging_weights_save_path is not None:
            if isinstance(file_path, str) and not file_path.startswith(("/", ".")):
                # if the file path is not absolute or relative to current working directory, save it in the log directory
                save_path = os.path.join(self.log_dir, file_path)
            else:
                save_path = file_path
            log.info(f"saving merging weights to {save_path}.")
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(merging_weights.detach().cpu(), save_path)

    def free_gpu_memory(self, module: LayerWiseMergedModel):
        module.pretrained_model.to("cpu")
        for model in module.task_vectors:
            model.to("cpu")
        del module
        gc.collect()
        torch.cuda.empty_cache()
        log.info(get_memory_usage("after freeing memory, the memory usage of GPU is:"))

    def update_datasets(self, datasets):
        """
        for evary epoch of local adamerging, we only use the data set corresponding to the model involved in the fusion
        """
        num_datasets = len(datasets)
        datasets_copy = datasets.copy()
        for i in range(num_datasets):
            datasets[i] = (
                datasets_copy[i]
                .union(datasets_copy[(i + 1) % num_datasets])
                .union(datasets_copy[(i - 1) % num_datasets])
            )
        return datasets

    def run(self, modelpool: Seq2SeqLMPool, **kwargs):
        """
        Run the Layer-Wise AdaMerging Algorithm.

        This method constructs the wrapped model and performs test-time adaptation if necessary.

        Args:
            modelpool (ModelPool): The model pool containing the pretrained and fine-tuned models.

        Returns:
            LayerWiseMergedModel: The merged model after test-time adaptation.
        """
        log.info("Fusing models using layer-wise adaptive merging.")
        self.modelpool = modelpool
        self.num_finetuned_models = len(modelpool.model_names)
        datasets = [{dataset} for dataset in modelpool.model_names]

        with self.profile("construct the wrapped model"):
            model_scheduler = ModelScheduler(self.configs, self.modelpool)

        if self.merging_weights_load_path is not None:
            # skip the test-time adaptation
            return module.merge_and_unload()
        else:
            for step_idx in tqdm(
                range(self.configs.gossip_max_steps),
                "Gossip merging",
                dynamic_ncols=True,
            ):
                datasets = self.update_datasets(datasets)
                log.info(f"Gossip merging step:, {step_idx}")
                for model_id in tqdm(
                    range(self.num_finetuned_models),
                    "local admerging",
                    dynamic_ncols=True,
                ):
                    if self.configs.gossip_skip_adamerging == True:
                        # skip adamerging, only merge
                        with self.profile("construct the local wrapped model"):
                            module = model_scheduler(model_id)
                        log.info(
                            f"skip adamerging, only merge ({modelpool.model_names[model_id]})"
                        )
                        model_scheduler.store_model(module.merge_weights(), model_id)
                        self.free_gpu_memory(module)
                    else:
                        with self.profile("construct the local wrapped model"):
                            module = model_scheduler(model_id)

                        if self.configs.improve_dataset == True:
                            log.info(
                                f"improved datasets, the datasets used in this local merging is {datasets[model_id]}"
                            )
                        else:
                            log.info(
                                f"unimproved datasets, the datasets used in this local merging is {modelpool.model_names}"
                            )
                        with self.profile("test-time adaptation"):
                            module = self.test_time_adaptation(
                                module, datasets[model_id]
                            )
                        # if self.configs.get("save_merging_weights", False):
                        #     self.save_merging_weights(
                        #         self.configs.save_merging_weights, module.merge_weight
                        #     )
                        model_scheduler.store_model(module.merge_weights(), model_id)
                        log.info(
                            get_memory_usage(
                                f"after local merging ({modelpool.model_names[model_id]}), the memory usage of GPU is:"
                            )
                        )
                        self.free_gpu_memory(
                            module
                        )  # simulate distributed GPU memory usage as much as possible

                model_scheduler.update_models()
                do_evaluation = False  # whether to do evaluation after each Gossip step
                if isinstance(self.configs.accuracy_test_interval, list):
                    if (step_idx + 1) in self.configs.accuracy_test_interval:
                        do_evaluation = True
                elif isinstance(self.configs.accuracy_test_interval, int):
                    if (
                        self.configs.accuracy_test_interval != 0
                        and (step_idx + 1) % self.configs.accuracy_test_interval == 0
                    ):
                        do_evaluation = True
                if do_evaluation:
                    self._program.evaluate_merged_model(
                        self._program.taskpool, model_scheduler.get_final_models()
                    )
                    model_scheduler.move_to("cpu")

        return model_scheduler.get_final_models()

    @functools.cache
    def get_shuffled_test_loader_iter(self, task: str) -> DataLoader:
        """
        Loader of test dataset for test-time adaptation. labels are not needed.

        Args:
            task (str): The name of the task.

        Returns:
            DataLoader: The data loader for the test dataset.
        """
        dataloader_kwargs = dict(self.dataloader_kwargs)
        dataloader_kwargs.update(dict(shuffle=True, collate_fn=default_data_collator))

        dataset = self.modelpool.load_test_dataset(task)
        loader = DataLoader(dataset, **dataloader_kwargs)

        if self.fabric is not None:
            loader = self.fabric.setup_dataloaders(loader)
        return iter(InfiniteDataLoader(loader))

    def compute_logits(
        self,
        module: Union[T5ForConditionalGeneration, LayerWiseMergedModel],
        batch,
        task: str,
    ) -> Tensor:
        """
        Compute the logits for the given images and task.

        Args:
            module: The model module.
            images (Tensor): The input images.
            task (str): The name of the task.

        Returns:
            Tensor: The computed logits.
        """
        input_ids: Tensor = batch["input_ids"]
        attention_mask: Tensor = batch["attention_mask"]

        # remove padding tokens from the input
        while attention_mask[:, -1].eq(0).all():
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]

        outputs = module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=torch.ones(
                input_ids.size(0), 1, dtype=torch.long, device=input_ids.device
            ),
        )
        logits = outputs.logits[:, 0, :]
        return logits

    def on_test_time_adaptation_start(self):
        """
        Something to do before the test-time adaptation starts. Such as setting up the task-specific heads.
        """
        pass

    def test_time_adaptation(self, module: LayerWiseMergedModel, datasets):
        """
        Perform test-time adaptation on the merged model.

        This method adapts the merging weights during test-time to improve performance.

        Args:
            module (LayerWiseMergedModel): The merged model.

        Returns:
            LayerWiseMergedModel: The adapted merged model.
        """
        self.on_test_time_adaptation_start()

        # configure optimizer
        optimizer = instantiate(self._optimizer, [module.merge_weight])
        module, optimizer = self.fabric.setup(module, optimizer)

        module.train()
        module.merge_weights()
        for step_idx in (
            pbar := tqdm(
                range(self.max_steps if not self.is_debug_mode else 1),
                ("[DEBUG MODE] " if self.is_debug_mode else "")
                + "AdaMerging Test-time adaptation",
                dynamic_ncols=True,
            )
        ):
            if self.variant == "mgda":
                total_loss = self._compute_gradients_using_mgda(module)
            else:
                total_loss = 0
                for task in self.modelpool.model_names:
                    with self.profile("data loading"):
                        batch = next(self.get_shuffled_test_loader_iter(task))
                    with self.profile("forward pass"):
                        logits = self.compute_logits(module, batch, task)
                        logits = logits.mean(dim=0, keepdim=True)
                        loss = entropy_loss(logits)
                        total_loss += loss
                    with self.profile("backward pass"):
                        self.fabric.backward(loss, retain_graph=True)

            with self.profile("optimizer step"):
                optimizer.step()
                optimizer.zero_grad()
            with self.profile("merging weights"):
                module.merge_weights()

            metrics = {
                "train/loss": total_loss.item(),
                "train/weight_max": module.merge_weight.max().item(),
                "train/weight_min": module.merge_weight.min().item(),
                "train/weight_mean": module.merge_weight.mean().item(),
            }
            self.fabric.log_dict(metrics, step=step_idx)
            pbar.set_postfix(metrics)

        self.print_profile_summary()
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
        return module

    def _compute_gradients_using_mgda(self, module: LayerWiseMergedModel):
        all_grads = []
        total_loss = 0
        # default behavior for first-order optimizers
        for task in self.modelpool.model_names:
            with self.profile("data loading"):
                batch = next(self.get_shuffled_test_loader_iter(task))
            with self.profile("forward pass"):
                logits = self.compute_logits(module, batch, task)
                logits = logits.mean(dim=0, keepdim=True)
                loss = entropy_loss(logits)
                total_loss += loss
            with self.profile("backward pass"):
                # self.fabric.backward(loss, retain_graph=True)
                _grads = torch.autograd.grad(
                    loss,
                    [module.merge_weight],
                    create_graph=False,
                    retain_graph=True,
                )
                all_grads.append(_grads[0].flatten().detach())
        sol, min_norm = MinNormSolver.find_min_norm_element(all_grads)
        if not isinstance(sol, torch.Tensor):
            sol = torch.from_numpy(sol)
        sol = sol.to(
            device=module.merge_weight.device,
            dtype=module.merge_weight.dtype,
        )
        grad = torch.stack(all_grads) * sol.view(-1, 1)
        module.merge_weight.grad = grad.sum(dim=0).view_as(module.merge_weight)
        return total_loss
