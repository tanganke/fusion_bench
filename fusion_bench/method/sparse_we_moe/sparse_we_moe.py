import logging
from abc import abstractmethod
from typing import cast

import lightning as L
import lightning.fabric.wrappers
import torch
from lightning.pytorch.profilers import SimpleProfiler
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from fusion_bench.compat.method import ModelFusionAlgorithm
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.models.sparse_we_moe import (
    SparseWeightEnsemblingMoE,
    SparseWeightEnsemblingMoE_ShardGate,
    _magnitude_prune,
    _module_magnitude_prune,
)
from fusion_bench.utils import timeit_context
from fusion_bench.utils.parameters import print_parameters

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


class SparseWeightEnsemblingMoEAlgorithm(ModelFusionAlgorithm):
    _fabric: L.Fabric = None
    modelpool: BaseModelPool = None

    def __init__(self, algorithm_config: DictConfig):
        """
        Initialize the SparseWeightEnsemblingMoEAlgorithm with the given configuration.

        Args:
            algorithm_config (DictConfig): The configuration for the algorithm.
        """
        super().__init__(algorithm_config)

        self.profiler = SimpleProfiler(
            self.config.get("cache_dir", "outputs"), "we_moe_profiler.txt"
        )

    @abstractmethod
    def load_checkpoint(self, model, checkpoint):
        """
        Load the checkpoint file.

        Args:
            model (nn.Module): The model to load the checkpoint into.
            checkpoint (str): The path to the checkpoint file.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, model, checkpoint):
        """
        Save the checkpoint file.

        Args:
            model (nn.Module): The model to save the checkpoint from.
            checkpoint (str): The path to the checkpoint file.
        """
        pass

    @abstractmethod
    def construct_moe_model(self) -> SparseWeightEnsemblingMoE:
        """
        Construct the Mixture of Experts model using the models in the model pool.

        Returns:
            SparseWeightEnsemblingMoE: The constructed Mixture of Experts model.
        """
        pass

    @abstractmethod
    def construct_moe_model_sharedgate(self) -> SparseWeightEnsemblingMoE_ShardGate:
        """
        Construct the Mixture of Experts model using the models in the model pool.

        Returns:
            SparseWeightEnsemblingMoE_ShardGate: The constructed Mixture of Experts model with shared gate.
        """
        pass

    def on_test_time_adaptation_start(self):
        """
        Hook that is called at the start of test-time adaptation.
        """
        pass

    @abstractmethod
    def get_shuffled_test_loader_iter(self, task: str) -> DataLoader:
        """
        Get an iterator for the shuffled test DataLoader for a specific task.

        Args:
            task (str): The task for which to get the DataLoader iterator.

        Returns:
            DataLoader: The DataLoader iterator for the specified task.
        """
        pass

    @abstractmethod
    def compute_logits(self, module, batch, task) -> Tensor:
        """
        Compute the logits for a given batch and task.

        Args:
            module (nn.Module): The model module.
            batch (Any): The input batch.
            task (str): The task for which to compute the logits.

        Returns:
            Tensor: The computed logits.
        """
        pass

    def dynamic_prune(self, module, prune_ratio):
        """
        Dynamically prune the parameters of a module based on the given prune ratio.

        Args:
            module (nn.Module): The module to prune.
            prune_ratio (float): The ratio of parameters to prune.
        """
        for param in module.parameters():
            if param.requires_grad:
                param.data = _magnitude_prune(param, prune_ratio)

    def l1_regularization(self, module, l1_lambda):
        """
        Compute the L1 regularization loss for a module.

        Args:
            module (nn.Module): The module for which to compute the L1 regularization loss.
            l1_lambda (float): The L1 regularization coefficient.

        Returns:
            Tensor: The L1 regularization loss.
        """
        l1_norm = sum(
            param.abs().sum() for param in module.parameters() if param.requires_grad
        )
        return l1_lambda * l1_norm

    def test_time_adaptation(self, module: SparseWeightEnsemblingMoE):
        """
        Perform test-time adaptation for the given module.

        Args:
            module (SparseWeightEnsemblingMoE): The module to adapt.

        Returns:
            SparseWeightEnsemblingMoE: The adapted module.
        """
        self.on_test_time_adaptation_start()

        # configure optimizer
        if self.config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                [p for p in module.parameters() if p.requires_grad], lr=self.config.lr
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        if self._fabric is not None:
            module, optimizer = self._fabric.setup(module, optimizer)

        module.train()

        if self.config.get("fast_dev_run", False):
            log.info("Running fast_dev_run, only one step")
            pbar = tqdm(
                range(1),
                "Test-time adaptation",
                dynamic_ncols=True,
            )
        else:
            pbar = tqdm(
                range(self.config.max_steps),
                "Test-time adaptation",
                dynamic_ncols=True,
            )

        for step_idx in pbar:
            if self.config.use_grad_accumulate:
                for task in self.modelpool.model_names:
                    with self.profiler.profile("data time"):
                        batch = next(self.get_shuffled_test_loader_iter(task))
                    with self.profiler.profile("forward pass"):
                        logits = self.compute_logits(module, batch, task)
                        assert (
                            logits.dim() == 2
                        ), f"Expected logits to be 2D, got {logits.dim()}"
                        loss = entropy_loss(logits)
                    # .backward() accumulates when .zero_grad() wasn't called
                    # this can save memory
                    with self.profiler.profile("backward pass"):
                        self._fabric.backward(loss, retain_graph=True)
            else:
                loss = 0
                for task in self.modelpool.model_names:
                    with self.profiler.profile("data time"):
                        batch = next(self.get_shuffled_test_loader_iter(task))
                    with self.profiler.profile("forward pass"):
                        logits = self.compute_logits(module, batch, task)
                        assert (
                            logits.dim() == 2
                        ), f"Expected logits to be 2D, got {logits.dim()}"
                        loss = loss + entropy_loss(logits)

                with self.profiler.profile("backward pass"):
                    self._fabric.backward(loss, retain_graph=True)

            with self.profiler.profile("optimizer step"):
                optimizer.step()
                optimizer.zero_grad()

        return module

    def construct_post_spare_gate_model(self, moe_model, gate_prune_ratio):
        """
        Construct a (post) sparse gated model.

        Args:
            moe_model (SparseWeightEnsemblingMoE): The Mixture of Experts model.
            gate_prune_ratio (float): The ratio of parameters to prune in the gate.

        Returns:
            SparseWeightEnsemblingMoE: The constructed (post) sparse gated model.
        """
        moe_encoder = moe_model.vision_model.encoder
        num_layers = len(moe_encoder.layers)
        for layer_idx in range(num_layers):
            gate = moe_encoder.layers[layer_idx].mlp.gate
            sparse_gate = _module_magnitude_prune(gate, gate_prune_ratio, layer_idx)
            moe_encoder.layers[layer_idx].mlp.gate = sparse_gate
        return moe_model

    def run(self, modelpool: BaseModelPool):
        """
        Run the SparseWeightEnsemblingMoEAlgorithm with the given model pool.

        Args:
            modelpool (BaseModelPool): The model pool to use for the algorithm.

        Returns:
            SparseWeightEnsemblingMoE: The final Mixture of Experts model.
        """
        log.info("Fusing models using WeightEnsembling Mixture of Experts modules.")
        self.modelpool = modelpool

        with timeit_context("upscaling models to a weight-ensembling MoE model"):
            if self.config.shared_gate:
                moe_model = self.construct_moe_model_sharedgate()
            else:
                moe_model = self.construct_moe_model()
            print_parameters(moe_model)

        if self.config.get("checkpoint", False):
            log.info(
                f"load checkpoint from {self.config.checkpoint}, test-time adaptation will be skipped."
            )
            self.load_checkpoint(moe_model, self.config.checkpoint)
        else:
            with self.profiler.profile("test-time adaptation"):
                moe_model = self.test_time_adaptation(moe_model)
            if self.config.get("save_checkpoint", False):
                log.info(f"save checkpoint to {self.config.save_checkpoint}")
                self.save_checkpoint(moe_model, self.config.save_checkpoint)

            if lightning.fabric.wrappers.is_wrapped(moe_model):
                moe_model = lightning.fabric.wrappers._unwrap_objects(moe_model)

        #  (post) sparse gate model
        if self.config.post_sparse_gate:
            moe_model = self.construct_post_spare_gate_model(
                moe_model, self.config.gate_prune_ratio
            )

        # enable sample-wise adaptation
        moe_model.batch_reduce = False
        print(self.profiler.summary())
        return moe_model
