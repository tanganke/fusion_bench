import logging
from abc import abstractmethod
from typing import cast  # noqa: F401

import lightning as L
import lightning.fabric.wrappers
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from fusion_bench.compat.method.base_algorithm import ModelFusionAlgorithm
from fusion_bench.compat.modelpool import ModelPool
from fusion_bench.mixins import SimpleProfilerMixin
from fusion_bench.models.we_moe import WeightEnsemblingMoE
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


class WeightEnsemblingMoEAlgorithm(
    ModelFusionAlgorithm,
    SimpleProfilerMixin,
):
    """
    Algorithm for fusing models using Weight Ensembling Mixture of Experts (MoE).

    This class provides methods for constructing the MoE model, performing test-time adaptation,
    and running the fusion process.

    Attributes:
        _fabric (L.Fabric): The fabric for distributed training.
        modelpool (ModelPool): The pool of models to be fused.
    """

    _fabric: L.Fabric = None
    modelpool: ModelPool = None

    def __init__(self, algorithm_config: DictConfig):
        """
        Initialize the WeightEnsemblingMoEAlgorithm with the given configuration.

        Args:
            algorithm_config (DictConfig): The configuration for the algorithm.
        """
        super().__init__(algorithm_config)

        if self._fabric is None and torch.cuda.is_available():
            self._fabric = L.Fabric(
                devices=self.config.get("devices", 1),
            )
            self._fabric.launch()
        else:
            assert "No CUDA device available."

    @abstractmethod
    def load_checkpoint(self, model, checkpoint):
        """
        Load the checkpoint file.

        Args:
            model: The model to load the checkpoint into.
            checkpoint: The checkpoint file to load.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, model, checkpoint):
        """
        Save the checkpoint file.

        Args:
            model: The model to save the checkpoint from.
            checkpoint: The checkpoint file to save.
        """
        pass

    @abstractmethod
    def construct_moe_model(self) -> WeightEnsemblingMoE:
        """
        Construct the Mixture of Experts model using the models in the model pool.

        Returns:
            WeightEnsemblingMoE: The constructed MoE model.
        """
        pass

    def on_test_time_adaptation_start(self):
        """
        Hook method called at the start of test-time adaptation.
        """
        pass

    @abstractmethod
    def get_shuffled_test_loader_iter(self, task: str) -> DataLoader:
        """
        Get an iterator for the shuffled test data loader for a specific task.

        Args:
            task (str): The task for which to get the test data loader.

        Returns:
            DataLoader: The shuffled test data loader iterator.
        """
        pass

    @abstractmethod
    def compute_logits(self, module, batch, task) -> Tensor:
        """
        Compute the logits for a given batch and task.

        Args:
            module: The model module to use for computing logits.
            batch: The batch of data.
            task: The task for which to compute logits.

        Returns:
            Tensor: The computed logits.
        """
        pass

    def test_time_adaptation(self, module: WeightEnsemblingMoE):
        """
        Perform test-time adaptation for the given module.

        Args:
            module (WeightEnsemblingMoE): The MoE module to adapt.

        Returns:
            WeightEnsemblingMoE: The adapted MoE module.
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
                    with self.profile("data time"):
                        batch = next(self.get_shuffled_test_loader_iter(task))
                    with self.profile("forward pass"):
                        logits = self.compute_logits(module, batch, task)
                        assert (
                            logits.dim() == 2
                        ), f"Expected logits to be 2D, got {logits.dim()}"
                        loss = entropy_loss(logits)
                    # .backward() accumulates when .zero_grad() wasn't called
                    # this can save memory
                    with self.profile("backward pass"):
                        self._fabric.backward(loss, retain_graph=True)
            else:
                loss = 0
                for task in self.modelpool.model_names:
                    with self.profile("data time"):
                        batch = next(self.get_shuffled_test_loader_iter(task))
                    with self.profile("forward pass"):
                        logits = self.compute_logits(module, batch, task)
                        assert (
                            logits.dim() == 2
                        ), f"Expected logits to be 2D, got {logits.dim()}"
                        loss = loss + entropy_loss(logits)
                with self.profile("backward pass"):
                    self._fabric.backward(loss, retain_graph=True)

            with self.profile("optimizer step"):
                optimizer.step()
                optimizer.zero_grad()

        return module

    def run(self, modelpool: ModelPool):
        """
        Run the WeightEnsemblingMoEAlgorithm to fuse models using Weight Ensembling Mixture of Experts.

        Args:
            modelpool (ModelPool): The pool of models to be fused.

        Returns:
            WeightEnsemblingMoE: The fused MoE model.
        """
        log.info("Fusing models using WeightEnsembling Mixture of Experts modules.")
        self.modelpool = modelpool

        with timeit_context("upscaling models to a weight-ensembling MoE model"):
            moe_model = self.construct_moe_model()
            print_parameters(moe_model)

        if self.config.get("checkpoint", False):
            log.info(
                f"load checkpoint from {self.config.checkpoint}, test-time adaptation will be skipped."
            )
            self.load_checkpoint(moe_model, self.config.checkpoint)
        else:
            with self.profile("test-time adaptation"):
                moe_model = self.test_time_adaptation(moe_model)
            if self.config.get("save_checkpoint", False):
                log.info(f"save checkpoint to {self.config.save_checkpoint}")
                self.save_checkpoint(moe_model, self.config.save_checkpoint)

            if lightning.fabric.wrappers.is_wrapped(moe_model):
                moe_model = lightning.fabric.wrappers._unwrap_objects(moe_model)

        # enable sample-wise adaptation
        moe_model.batch_reduce = False
        self.print_profile_summary()
        return moe_model
