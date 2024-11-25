"""
This is an experimental implementation of the Layer-Wise AdaMerging Algorithm for GPT-2 models.
The efficiency of the algorithm is not guaranteed, and it may not work as expected.
"""

import functools
import logging
import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union, cast  # noqa: F401

import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import GPT2ForSequenceClassification, GPT2Model
from transformers.data import default_data_collator

from fusion_bench.method import BaseAlgorithm
from fusion_bench.method.simple_average import simple_average
from fusion_bench.mixins.lightning_fabric import LightningFabricMixin
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import GPT2ForSequenceClassificationPool
from fusion_bench.models.wrappers.layer_wise_fusion import (
    LayerWiseMergedModel,
    get_layer_wise_weights,
)
from fusion_bench.utils.data import InfiniteDataLoader, load_tensor_from_file
from fusion_bench.utils.instantiate import instantiate

from .entropy_loss import entropy_loss
from .min_norm_solvers import MinNormSolver
from .utils import get_memory_usage

log = logging.getLogger(__name__)


class GPT2LayerWiseAdaMergingAlgorithm(
    BaseAlgorithm,
    LightningFabricMixin,
    SimpleProfilerMixin,
):
    scores: Dict[str, nn.Linear] = None

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
        super().__init__(**kwargs)

    @torch.no_grad()
    def construct_layer_wise_merged_model(
        self, modelpool: GPT2ForSequenceClassificationPool
    ):
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
        pretrained_model: GPT2Model = modelpool.load_model("_pretrained_")
        finetuned_models: List[GPT2Model] = [
            modelpool.load_model(name) for name in modelpool.model_names
        ]

        # initialize layer-wise weights using the provided configuration `init_values` or load from file if `weights` is provided
        if self.merging_weights_load_path is None:
            layer_wise_weight = get_layer_wise_weights(
                num_models=len(modelpool.model_names),
                num_layers=len(
                    tuple(
                        filter(lambda p: p.requires_grad, pretrained_model.parameters())
                    )
                ),
                init_values=self.init_values,
            )
        else:
            if isinstance(self.merging_weights_load_path, str):
                # load the merging weights from a file
                layer_wise_weight = load_tensor_from_file(
                    self.merging_weights_load_path
                )
            else:
                raise ValueError(
                    f"Unsupported weights format: {self.merging_weights_load_path}"
                )

        module = LayerWiseMergedModel(
            layer_wise_weight=layer_wise_weight,
            pretrained_model=pretrained_model,
            finetuned_models=finetuned_models,
            clamp_weights=self.clamp_weights,
            tie_weights=self.tie_weights,
            strict=self.strict,
        )
        print(f"{layer_wise_weight.size()=}, {layer_wise_weight.numel()=}")
        return module

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

    def run(self, modelpool: GPT2ForSequenceClassificationPool, **kwargs):
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

        with self.profile("construct the wrapped model"):
            module = self.construct_layer_wise_merged_model(modelpool)

        if self.merging_weights_load_path is not None:
            # skip the test-time adaptation
            return module.merge_and_unload()
        else:
            with self.profile("test-time adaptation"):
                module = self.test_time_adaptation(module)
            if self.merging_weights_save_path is not None:
                self.save_merging_weights(
                    self.merging_weights_save_path, module.merge_weight
                )
            return module.merge_and_unload()

    def on_test_time_adaptation_start(self):
        """
        Something to do before the test-time adaptation starts. Such as setting up the task-specific heads.
        """
        self.scores = {}
        for model_name in self.modelpool.model_names:
            score = cast(
                GPT2ForSequenceClassification,
                self.modelpool.load_classifier(model_name),
            ).score.requires_grad_(False)
            score = score.to(self.fabric.device)
            self.scores[model_name] = score

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

    def compute_logits(self, module: GPT2Model, batch, task: str) -> Tensor:
        """
        Compute the logits for the given images and task.

        Args:
            module: The model module.
            images (Tensor): The input images.
            task (str): The name of the task.

        Returns:
            Tensor: The computed logits.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_size, _ = input_ids.shape[:2]
        pad_token_id = 50256

        transformer_outputs = module(
            input_ids,
            past_key_values=None,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
        )
        hidden_states = transformer_outputs[0]
        logits = self.scores[task](hidden_states)

        sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(logits.device)

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        assert pooled_logits.dim() == 2
        return pooled_logits

    def test_time_adaptation(self, module: LayerWiseMergedModel):
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

        log.info(get_memory_usage(f"after adamerging, the memory usage of GPU is:"))
        self.print_profile_summary()
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
