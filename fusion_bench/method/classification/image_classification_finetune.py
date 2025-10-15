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
from torch.utils.data import DataLoader
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


@auto_register_config
class ImageClassificationFineTuning(BaseAlgorithm):
    def __init__(
        self,
        max_epochs: Optional[int],
        max_steps: Optional[int],
        label_smoothing: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        dataloader_kwargs: DictConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert (max_epochs is None) or (
            max_steps is None or max_steps < 0
        ), "Only one of max_epochs or max_steps should be set."
        self.training_interval = "epoch" if max_epochs is not None else "step"
        if self.training_interval == "epoch":
            self.max_steps = -1
        log.info(f"Training interval: {self.training_interval}")
        log.info(f"Max epochs: {max_epochs}, max steps: {max_steps}")

    def run(self, modelpool: ResNetForImageClassificationPool):
        # load model and dataset
        model = modelpool.load_pretrained_or_first_model()
        assert isinstance(model, nn.Module), "Loaded model is not a nn.Module."

        assert (
            len(modelpool.train_dataset_names) == 1
        ), "Exactly one training dataset is required."
        self.dataset_name = dataset_name = modelpool.train_dataset_names[0]
        num_classes = get_num_classes(dataset_name)
        train_dataset = modelpool.load_train_dataset(dataset_name)
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
                "acc@5": Accuracy(task="multiclass", num_classes=num_classes, top_k=5),
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
            ],
            logger=TensorBoardLogger(
                save_dir=log_dir,
                name="",
            ),
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
            )
        return model

    def get_dataloader(self, dataset, stage: str):
        assert stage in ["train", "val", "test"], f"Invalid stage: {stage}"
        dataloader_kwargs = dict(self.dataloader_kwargs)
        if "shuffle" not in dataloader_kwargs:
            dataloader_kwargs["shuffle"] = stage == "train"
        return DataLoader(dataset, **dataloader_kwargs)


@auto_register_config
class ImageClassificationFineTuning_Test(BaseAlgorithm):
    def __init__(self, checkpoint_path: str, dataloader_kwargs: DictConfig, **kwargs):
        super().__init__(**kwargs)

    def run(self, modelpool: BaseModelPool):
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
                    "acc@5": Accuracy(
                        task="multiclass", num_classes=num_classes, top_k=5
                    ),
                },
            )
        else:
            lit_module = ERM_LitModule.load_from_checkpoint(
                checkpoint_path=self.checkpoint_path,
                model=model,
                metrics={
                    "acc@1": Accuracy(task="multiclass", num_classes=num_classes),
                    "acc@5": Accuracy(
                        task="multiclass", num_classes=num_classes, top_k=5
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
        assert stage in ["train", "val", "test"], f"Invalid stage: {stage}"
        dataloader_kwargs = dict(self.dataloader_kwargs)
        if "shuffle" not in dataloader_kwargs:
            dataloader_kwargs["shuffle"] = stage == "train"
        return DataLoader(dataset, **dataloader_kwargs)
