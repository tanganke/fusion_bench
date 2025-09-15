from typing import Optional

import lightning as L
import lightning.pytorch.callbacks as pl_callbacks
import torch
from lit_learn.lit_modules import ERM_LitModule
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy

from fusion_bench import BaseAlgorithm, BaseModelPool, auto_register_config, instantiate
from fusion_bench.dataset import CLIPDataset
from fusion_bench.modelpool import ResNetForImageClassificationPool
from fusion_bench.tasks.clip_classification import get_classnames_and_templates


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
        print(f"Training interval: {self.training_interval}")
        print(f"Max epochs: {max_epochs}, max steps: {max_steps}")

    def run(self, modelpool: ResNetForImageClassificationPool):
        # load model and dataset
        model = modelpool.load_pretrained_or_first_model()
        assert isinstance(model, nn.Module), "Loaded model is not a nn.Module."

        train_dataset = modelpool.load_train_dataset(modelpool.train_dataset_names[0])
        train_dataset = CLIPDataset(
            train_dataset, processor=modelpool.load_processor(stage="train")
        )
        train_loader = self.get_dataloader(train_dataset, stage="train")
        if modelpool.has_val_dataset:
            val_dataset = modelpool.load_val_dataset(modelpool.val_dataset_names[0])
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
        print(f"optimizer:\n{optimizer}")

        lit_module = ERM_LitModule(
            model,
            optimizer,
            objective=nn.CrossEntropyLoss(label_smoothing=self.label_smoothing),
            metrics={
                "acc@1": Accuracy(task="multiclass", num_classes=1000),
                "acc@5": Accuracy(task="multiclass", num_classes=1000, top_k=5),
            },
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
        )

        trainer.fit(
            lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
        return lit_module.model

    def get_dataloader(self, dataset, stage: str):
        assert stage in ["train", "val", "test"], f"Invalid stage: {stage}"
        dataloader_kwargs = dict(self.dataloader_kwargs)
        if "shuffle" not in dataloader_kwargs:
            dataloader_kwargs["shuffle"] = stage == "train"
        return DataLoader(dataset, **dataloader_kwargs)


@auto_register_config
class ImageClassificationFineTuning_Test(BaseAlgorithm):
    def __init__(self, dataloader_kwargs: DictConfig, **kwargs):
        super().__init__(**kwargs)

    def run(self, modelpool: BaseModelPool):
        assert (
            modelpool.has_val_dataset or modelpool.has_test_dataset
        ), "No validation or test dataset found in the model pool."

        # load model and dataset
        model = modelpool.load_pretrained_or_first_model()
        assert isinstance(model, nn.Module), "Loaded model is not a nn.Module."

        if modelpool.has_test_dataset:
            dataset = modelpool.load_test_dataset(modelpool.test_dataset_names[0])
            dataset = CLIPDataset(
                dataset, processor=modelpool.load_processor(stage="test")
            )
        else:
            dataset = modelpool.load_val_dataset(modelpool.val_dataset_names[0])
            dataset = CLIPDataset(
                dataset, processor=modelpool.load_processor(stage="test")
            )

        test_loader = self.get_dataloader(dataset, stage="test")

        if self.checkpoint_path is None:
            lit_module = ERM_LitModule(
                model,
                metrics={
                    "acc@1": Accuracy(task="multiclass", num_classes=1000),
                    "acc@5": Accuracy(task="multiclass", num_classes=1000, top_k=5),
                },
            )
        else:
            lit_module = ERM_LitModule.load_from_checkpoint(
                checkpoint_path=self.checkpoint_path,
                model=model,
                metrics={
                    "acc@1": Accuracy(task="multiclass", num_classes=1000),
                    "acc@5": Accuracy(task="multiclass", num_classes=1000, top_k=5),
                },
            )

        trainer = L.Trainer(devices=1, num_nodes=1, logger=False)

        test_metrics = trainer.test(lit_module, dataloaders=test_loader)
        print(f"Test metrics: {test_metrics}")
        return model

    def get_dataloader(self, dataset, stage: str):
        assert stage in ["train", "val", "test"], f"Invalid stage: {stage}"
        dataloader_kwargs = dict(self.dataloader_kwargs)
        if "shuffle" not in dataloader_kwargs:
            dataloader_kwargs["shuffle"] = stage == "train"
        return DataLoader(dataset, **dataloader_kwargs)
