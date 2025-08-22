import functools
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Union, cast  # noqa: F401

import lightning
import lightning as L
import lightning.fabric.wrappers
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import T5ForConditionalGeneration
from transformers.data import default_data_collator

from fusion_bench.method import BaseAlgorithm
from fusion_bench.method.task_arithmetic.task_arithmetic import task_arithmetic_merge
from fusion_bench.mixins import (
    LightningFabricMixin,
    SimpleProfilerMixin,
    auto_register_config,
)
from fusion_bench.modelpool import Seq2SeqLMPool
from fusion_bench.models.we_moe import WeightEnsemblingMoE
from fusion_bench.utils import print_parameters, timeit_context
from fusion_bench.utils.data import InfiniteDataLoader, load_tensor_from_file
from fusion_bench.utils.instantiate_utils import instantiate
from fusion_bench.utils.parameters import print_parameters

from .entropy_loss import entropy_loss
from .utils import get_memory_usage

log = logging.getLogger(__name__)


@auto_register_config
class FlanT5WeightEnsemblingMoEAlgorithm(
    LightningFabricMixin,
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    """
    FlanT5WeightEnsemblingMoEAlgorithm is a class that implements the WeightEnsemblingMoEAlgorithm
    for FlanT5 models. It extends the WeightEnsemblingMoEAlgorithm and CLIPClassificationMixin classes.

    Attributes:
        modelpool (Seq2SeqLMPool): The model pool containing the FlanT5 models.
    """

    modelpool: Seq2SeqLMPool = None

    def __init__(
        self,
        checkpoint: bool = False,
        save_checkpoint: bool = False,
        router_hidden_layers: int = 2,
        init_lambda: float = 0.3,
        batch_reduce: bool = True,
        lr: float = 1e-4,
        optimizer: str = "adam",
        devices: int = 1,
        batch_size: int = 16,
        num_workers: int = 0,
        max_steps: int = 1000,
        use_grad_accumulate: bool = True,
        fast_dev_run: bool = False,
        **kwargs,
    ):
        """
        Initialize the WeightEnsemblingMoEAlgorithm with the given configuration.

        Args:
            algorithm_config (DictConfig): The configuration for the algorithm.
        """
        super().__init__(**kwargs)

    def construct_moe_model(self):
        """
        Construct the Mixture of Experts (MoE) model using the models in the model pool.

        Returns:
            WeightEnsemblingMoE: The constructed MoE model.
        """
        base_model = self.modelpool.load_model("_pretrained_")
        expert_models = [
            self.modelpool.load_model(name) for name in self.modelpool.model_names
        ]

        # Merge the models using task arithmetic
        moe_model = task_arithmetic_merge(
            # This function modifies the model in place, so we need to pass a deepcopy
            deepcopy(base_model),
            expert_models,
            scaling_factor=self.init_lambda,
        ).requires_grad_(False)

        print(base_model)

        # Up-scale MLP modules
        num_layer = 12
        encoder_mlp_index = 1
        base_encoder = base_model.encoder
        moe_encoder = moe_model.encoder
        expert_encoders = [m.encoder for m in expert_models]

        for layer_idx in range(num_layer):
            base_mlp = (
                base_encoder.block[layer_idx].layer[encoder_mlp_index].DenseReluDense
            )
            expert_mlps = [
                e.block[layer_idx].layer[encoder_mlp_index].DenseReluDense
                for e in expert_encoders
            ]

            moe_encoder.block[layer_idx].layer[encoder_mlp_index].DenseReluDense = (
                WeightEnsemblingMoE(
                    hidden_size=base_encoder.config.hidden_size,
                    base_model=base_mlp,
                    expert_models=expert_mlps,
                    init_lambda=self.init_lambda,
                    batch_first=True,
                    router_hidden_layers=self.router_hidden_layers,
                    batch_reduce=self.batch_reduce,
                )
            )

        decoder_mlp_index = 2
        base_decoder = base_model.decoder
        moe_decoder = moe_model.decoder
        expert_decoders = [m.decoder for m in expert_models]

        for layer_idx in range(num_layer):
            base_mlp = (
                base_decoder.block[layer_idx].layer[decoder_mlp_index].DenseReluDense
            )
            expert_mlps = [
                e.block[layer_idx].layer[decoder_mlp_index].DenseReluDense
                for e in expert_decoders
            ]

            moe_decoder.block[layer_idx].layer[decoder_mlp_index].DenseReluDense = (
                WeightEnsemblingMoE(
                    hidden_size=base_decoder.config.hidden_size,
                    base_model=base_mlp,
                    expert_models=expert_mlps,
                    init_lambda=self.init_lambda,
                    batch_first=True,
                    router_hidden_layers=self.router_hidden_layers,
                    batch_reduce=self.batch_reduce,
                )
            )

        print(moe_model)
        return moe_model

    @functools.cache
    def get_shuffled_test_loader_iter(self, task: str) -> DataLoader:
        """
        Loader of test dataset for test-time adaptation. labels are not needed.

        Args:
            task (str): The name of the task.

        Returns:
            DataLoader: The data loader for the test dataset.
        """
        # dataloader_kwargs = dict(self.dataloader_kwargs)
        # dataloader_kwargs.update(dict(shuffle=True, collate_fn=default_data_collator))

        dataset = self.modelpool.load_test_dataset(task)
        log.info("get_shuffled_test_loader_iter")
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=default_data_collator,
        )
        # loader = DataLoader(dataset, **dataloader_kwargs)
        if self.fabric is not None:
            loader = self.fabric.setup_dataloaders(loader)
        return iter(InfiniteDataLoader(loader))

    def compute_logits(
        self,
        module: Union[T5ForConditionalGeneration],
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

    def test_time_adaptation(self, module):
        """
        Perform test-time adaptation for the given module.

        Args:
            module (WeightEnsemblingMoE): The MoE module to adapt.

        Returns:
            WeightEnsemblingMoE: The adapted MoE module.
        """
        self.on_test_time_adaptation_start()

        # configure optimizer
        if self.optimizer == "adam":
            print([name for name, p in module.named_parameters() if p.requires_grad])
            optimizer = torch.optim.Adam(
                [p for p in module.parameters() if p.requires_grad], lr=self.lr
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        module, optimizer = self.fabric.setup(module, optimizer)

        module.train()
        # module.merge_weights()
        for step_idx in (
            pbar := tqdm(
                range(self.max_steps if not self.is_debug_mode else 1),
                ("[DEBUG MODE] " if self.is_debug_mode else "")
                + "WEMoE Test-time adaptation",
                dynamic_ncols=True,
            )
        ):
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

            metrics = {
                "train/loss": total_loss.item(),
            }
            self.fabric.log_dict(metrics, step=step_idx)
            pbar.set_postfix(metrics)

        log.info(get_memory_usage(f"after adamerging, the memory usage of GPU is:"))
        self.print_profile_summary()
        return module

    def on_test_time_adaptation_start(self):
        """
        Something to do before the test-time adaptation starts. Such as setting up the task-specific heads.
        """
        pass

    def run(self, modelpool: Seq2SeqLMPool, **kwargs):
        """
        Run the WeightEnsemblingMoEAlgorithm to fuse models using Weight Ensembling Mixture of Experts.

        Args:
            modelpool (ModelPool): The pool of models to be fused.

        Returns:
            WeightEnsemblingMoE: The fused MoE model.
        """
        log.info("Fusing models using layer-wise adaptive merging.")
        self.modelpool = modelpool

        with timeit_context("upscaling models to a weight-ensembling MoE model"):
            moe_model = self.construct_moe_model()
            print_parameters(moe_model)

        if self.checkpoint != False:
            log.info(
                f"load checkpoint from {self.checkpoint}, test-time adaptation will be skipped."
            )
            self.load_checkpoint(moe_model, self.checkpoint)
        else:
            with self.profile("test-time adaptation"):
                moe_model = self.test_time_adaptation(moe_model)
            if self.save_checkpoint != False:
                log.info(f"save checkpoint to {self.save_checkpoint}")
                self.save_checkpoint(moe_model, self.save_checkpoint)

            if lightning.fabric.wrappers.is_wrapped(moe_model):
                moe_model = lightning.fabric.wrappers._unwrap_objects(moe_model)

        # enable sample-wise adaptation
        moe_model.batch_reduce = False
        self.print_profile_summary()
        return moe_model
