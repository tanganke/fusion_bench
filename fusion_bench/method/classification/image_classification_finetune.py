"""Image Classification Fine-tuning Module.

This module provides algorithms for fine-tuning and evaluating image classification models
using PyTorch Lightning.
"""

import os
from typing import Optional

import lightning as L
import lightning.pytorch.callbacks as pl_callbacks
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning_utilities.core.rank_zero import rank_zero_only
from lit_learn.lit_modules import ERM_LitModule
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import Accuracy

from fusion_bench import (
    BaseAlgorithm,
    BaseModelPool,
    RuntimeConstants,
    auto_register_config,
    get_rankzero_logger,
    instantiate,
)
from fusion_bench.dataset import CLIPDataset
from fusion_bench.modelpool import ResNetForImageClassificationPool
from fusion_bench.tasks.clip_classification import get_num_classes

log = get_rankzero_logger(__name__)


def _get_base_model_name(model) -> Optional[str]:
    if hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
        return model.config._name_or_path
    else:
        return None


@auto_register_config
class ImageClassificationFineTuning(BaseAlgorithm):
    """Fine-tuning algorithm for image classification models.

    This class implements end-to-end fine-tuning for image classification tasks using PyTorch Lightning.
    It supports both epoch-based and step-based training with configurable optimizers, learning rate
    schedulers, and data loaders.

    Args:
        max_epochs (Optional[int]): Maximum number of training epochs. Mutually exclusive with max_steps.
        max_steps (Optional[int]): Maximum number of training steps. Mutually exclusive with max_epochs.
        label_smoothing (float): Label smoothing factor for cross-entropy loss (0.0 = no smoothing).
        optimizer (DictConfig): Configuration for the optimizer (e.g., Adam, SGD).
        lr_scheduler (DictConfig): Configuration for the learning rate scheduler.
        dataloader_kwargs (DictConfig): Additional keyword arguments for DataLoader construction.
        **kwargs: Additional arguments passed to the base class.

    Raises:
        AssertionError: If both max_epochs and max_steps are provided.

    Example:
        ```python
        >>> config = {
        ...     'max_epochs': 10,
        ...     'max_steps': None,
        ...     'label_smoothing': 0.1,
        ...     'optimizer': {'_target_': 'torch.optim.Adam', 'lr': 0.001},
        ...     'lr_scheduler': {'_target_': 'torch.optim.lr_scheduler.StepLR', 'step_size': 5},
        ...     'dataloader_kwargs': {'batch_size': 32, 'num_workers': 4}
        ... }
        >>> algorithm = ImageClassificationFineTuning(**config)
        ```
    """

    def __init__(
        self,
        max_epochs: Optional[int],
        max_steps: Optional[int],
        training_data_ratio: Optional[float],
        label_smoothing: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        dataloader_kwargs: DictConfig,
        save_top_k: int,
        save_interval: int,
        save_on_train_epoch_end: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert (max_epochs is None or max_epochs < 0) or (
            max_steps is None or max_steps < 0
        ), "Only one of max_epochs or max_steps should be set."
        self.training_interval = (
            "epoch" if max_epochs is not None and max_epochs > 0 else "step"
        )
        if self.training_interval == "epoch":
            self.max_steps = -1
        log.info(f"Training interval: {self.training_interval}")
        log.info(f"Max epochs: {max_epochs}, max steps: {max_steps}")

    def run(self, modelpool: ResNetForImageClassificationPool):
        """Execute the fine-tuning process on the provided model pool.

        This method performs the complete fine-tuning workflow:
        1. Loads the pretrained model from the model pool
        2. Prepares training and validation datasets
        3. Configures optimizer and learning rate scheduler
        4. Sets up Lightning trainer with appropriate callbacks
        5. Executes the training process
        6. Saves the final fine-tuned model
        """
        # load model and dataset
        model = modelpool.load_pretrained_or_first_model()
        base_model_name = _get_base_model_name(model)

        assert isinstance(model, nn.Module), "Loaded model is not a nn.Module."

        assert (
            len(modelpool.train_dataset_names) == 1
        ), "Exactly one training dataset is required."
        self.dataset_name = dataset_name = modelpool.train_dataset_names[0]
        num_classes = get_num_classes(dataset_name)
        log.info(f"Number of classes for dataset {dataset_name}: {num_classes}")
        train_dataset = modelpool.load_train_dataset(dataset_name)
        log.info(f"Training dataset size: {len(train_dataset)}")
        if self.training_data_ratio is not None and 0 < self.training_data_ratio < 1:
            train_dataset, _ = random_split(
                train_dataset,
                lengths=[self.training_data_ratio, 1 - self.training_data_ratio],
            )
            log.info(
                f"Using {len(train_dataset)} samples for training after applying training_data_ratio={self.training_data_ratio}."
            )
        train_dataset = CLIPDataset(
            train_dataset, processor=modelpool.load_processor(stage="train")
        )
        train_loader = self.get_dataloader(train_dataset, stage="train")
        if modelpool.has_val_dataset:
            val_dataset = modelpool.load_val_dataset(dataset_name)
            val_dataset = CLIPDataset(
                val_dataset, processor=modelpool.load_processor(stage="val")
            )
            val_loader = self.get_dataloader(val_dataset, stage="val")
        else:
            val_loader = None

        # configure optimizer
        optimizer = instantiate(self.optimizer, params=model.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = instantiate(self.lr_scheduler, optimizer=optimizer)
            optimizer = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": self.training_interval,
                    "frequency": 1,
                },
            }
        log.info(f"optimizer:\n{optimizer}")

        lit_module = ERM_LitModule(
            model,
            optimizer,
            objective=nn.CrossEntropyLoss(label_smoothing=self.label_smoothing),
            metrics={
                "acc@1": Accuracy(task="multiclass", num_classes=num_classes),
                f"acc@{min(5,num_classes)}": Accuracy(
                    task="multiclass",
                    num_classes=num_classes,
                    top_k=min(5, num_classes),
                ),
            },
        )

        log_dir = (
            self._program.path.log_dir
            if self._program is not None
            else "outputs/lightning_logs"
        )
        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            max_steps=self.max_steps,
            accelerator="auto",
            devices="auto",
            callbacks=[
                pl_callbacks.LearningRateMonitor(logging_interval="step"),
                pl_callbacks.DeviceStatsMonitor(),
                pl_callbacks.ModelCheckpoint(
                    save_top_k=self.save_top_k,
                    every_n_train_steps=(
                        self.save_interval if self.training_interval == "step" else None
                    ),
                    every_n_epochs=(
                        self.save_interval
                        if self.training_interval == "epoch"
                        else None
                    ),
                    save_on_train_epoch_end=self.save_on_train_epoch_end,
                    save_last=True,
                ),
            ],
            logger=TensorBoardLogger(save_dir=log_dir, name="", version=""),
            fast_dev_run=RuntimeConstants.debug,
        )

        trainer.fit(
            lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
        model = lit_module.model
        if rank_zero_only.rank == 0:
            log.info(f"Saving the final model to {log_dir}/raw_checkpoints/final")
            modelpool.save_model(
                model,
                path=os.path.join(
                    trainer.log_dir if trainer.log_dir is not None else log_dir,
                    "raw_checkpoints",
                    "final",
                ),
                algorithm_config=self.config,
                description=f"Fine-tuned ResNet model on dataset {dataset_name}.",
                base_model=base_model_name,
            )
        return model

    def get_dataloader(self, dataset, stage: str):
        """Create a DataLoader for the specified dataset and training stage.

        Constructs a PyTorch DataLoader with stage-appropriate configurations:
        - Training stage: shuffling enabled by default
        - Validation/test stages: shuffling disabled by default

        Args:
            dataset: The dataset to wrap in a DataLoader.
            stage (str): Training stage, must be one of "train", "val", or "test".
                Determines default shuffling behavior.

        Returns:
            DataLoader: Configured DataLoader for the given dataset and stage.
        """
        assert stage in ["train", "val", "test"], f"Invalid stage: {stage}"
        dataloader_kwargs = dict(self.dataloader_kwargs)
        if "shuffle" not in dataloader_kwargs:
            dataloader_kwargs["shuffle"] = stage == "train"
        return DataLoader(dataset, **dataloader_kwargs)


@auto_register_config
class ImageClassificationFineTuning_Test(BaseAlgorithm):
    """Test/evaluation algorithm for fine-tuned image classification models.

    This class implements model evaluation on test or validation datasets using PyTorch Lightning.
    It can either evaluate a model directly or load a model from a checkpoint before evaluation.
    The evaluation computes standard classification metrics including top-1 and top-5 accuracy.

    Args:
        checkpoint_path (str): Path to the model checkpoint file. If None, uses the model
            directly from the model pool without loading from checkpoint.
        dataloader_kwargs (DictConfig): Additional keyword arguments for DataLoader construction.
        **kwargs: Additional arguments passed to the base class.

    Example:
        ```python
        >>> config = {
        ...     'checkpoint_path': '/path/to/model/checkpoint.ckpt',
        ...     'dataloader_kwargs': {'batch_size': 64, 'num_workers': 4}
        ... }
        >>> test_algorithm = ImageClassificationFineTuning_Test(**config)
        ```
    """

    def __init__(self, checkpoint_path: str, dataloader_kwargs: DictConfig, **kwargs):
        super().__init__(**kwargs)

    def run(self, modelpool: ResNetForImageClassificationPool):
        """Execute model evaluation on the provided model pool's test/validation dataset.

        This method performs the complete evaluation workflow:
        1. Loads the model from the model pool (pretrained or first available)
        2. Prepares the test or validation dataset (prioritizes test if both available)
        3. Sets up the Lightning module with appropriate metrics (top-1 and top-5 accuracy)
        4. Loads from checkpoint if specified, otherwise uses the model directly
        5. Executes the evaluation using Lightning trainer
        6. Logs and returns the test metrics
        """
        assert (
            modelpool.has_val_dataset or modelpool.has_test_dataset
        ), "No validation or test dataset found in the model pool."

        # load model and dataset
        model = modelpool.load_pretrained_or_first_model()
        assert isinstance(model, nn.Module), "Loaded model is not a nn.Module."

        if modelpool.has_test_dataset:
            assert (
                len(modelpool.test_dataset_names) == 1
            ), "Exactly one test dataset is required."
            self.dataset_name = dataset_name = modelpool.test_dataset_names[0]
            dataset = modelpool.load_test_dataset(dataset_name)
            dataset = CLIPDataset(
                dataset, processor=modelpool.load_processor(stage="test")
            )
        else:
            assert (
                len(modelpool.val_dataset_names) == 1
            ), "Exactly one validation dataset is required."
            self.dataset_name = dataset_name = modelpool.val_dataset_names[0]
            dataset = modelpool.load_val_dataset(dataset_name)
            dataset = CLIPDataset(
                dataset, processor=modelpool.load_processor(stage="test")
            )
        num_classes = get_num_classes(dataset_name)

        test_loader = self.get_dataloader(dataset, stage="test")

        if self.checkpoint_path is None:
            lit_module = ERM_LitModule(
                model,
                metrics={
                    "acc@1": Accuracy(task="multiclass", num_classes=num_classes),
                    f"acc@{min(5,num_classes)}": Accuracy(
                        task="multiclass",
                        num_classes=num_classes,
                        top_k=min(5, num_classes),
                    ),
                },
            )
        else:
            lit_module = ERM_LitModule.load_from_checkpoint(
                checkpoint_path=self.checkpoint_path,
                model=model,
                metrics={
                    "acc@1": Accuracy(task="multiclass", num_classes=num_classes),
                    f"acc@{min(5,num_classes)}": Accuracy(
                        task="multiclass",
                        num_classes=num_classes,
                        top_k=min(5, num_classes),
                    ),
                },
            )

        trainer = L.Trainer(
            devices=1, num_nodes=1, logger=False, fast_dev_run=RuntimeConstants.debug
        )

        test_metrics = trainer.test(lit_module, dataloaders=test_loader)
        log.info(f"Test metrics: {test_metrics}")
        return model

    def get_dataloader(self, dataset, stage: str):
        """Create a DataLoader for the specified dataset and evaluation stage.

        Constructs a PyTorch DataLoader with stage-appropriate configurations for evaluation.
        Similar to the training version but typically used for test/validation datasets.

        Args:
            dataset: The dataset to wrap in a DataLoader.
            stage (str): Evaluation stage, must be one of "train", "val", or "test".
                Determines default shuffling behavior (disabled for non-train stages).

        Returns:
            DataLoader: Configured DataLoader for the given dataset and stage.
        """
        assert stage in ["train", "val", "test"], f"Invalid stage: {stage}"
        dataloader_kwargs = dict(self.dataloader_kwargs)
        if "shuffle" not in dataloader_kwargs:
            dataloader_kwargs["shuffle"] = stage == "train"
        return DataLoader(dataset, **dataloader_kwargs)
