R"""
This script is used to train a multi-task learning (MTL) model on the NYUv2 dataset.
"""

import importlib
import logging
import os
import sys
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
)
from omegaconf import DictConfig, OmegaConf
from rich import print as rich_print
from rich.syntax import Syntax
from torch import nn
from torch.utils.data import DataLoader

from fusion_bench.dataset.nyuv2 import NYUv2
from fusion_bench.models.nyuv2.aspp import DeepLabHead
from fusion_bench.models.nyuv2.lightning_module import NYUv2MTLModule as _NYUv2MTLModule
from fusion_bench.models.nyuv2.resnet_dilated import resnet_dilated
from fusion_bench.utils.parameters import print_parameters

log = logging.getLogger(__name__)


class NYUv2MTLModule(_NYUv2MTLModule):
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )
        return [optimizer], [lr_scheduler]


class Program:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def run(self):
        cfg = self.cfg

        self.load_module()
        self.load_datasets()

        trainer = L.Trainer(
            **OmegaConf.to_container(cfg.trainer, resolve=True),
            callbacks=[
                DeviceStatsMonitor(),
                LearningRateMonitor(logging_interval="step"),
                RichModelSummary(max_depth=1),
                ModelCheckpoint(save_last=True),
            ],
        )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

        trainer.fit(
            self.module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=cfg.ckpt_path,
        )

    def load_module(self):
        cfg = self.cfg

        log.info("create encoder")
        encoder = resnet_dilated("resnet50")
        log.info("create decoders")
        decoders = nn.ModuleDict(
            {
                task: DeepLabHead(2048, NYUv2.num_out_channels[task])
                for task in cfg.tasks
            }
        )

        module = NYUv2MTLModule(
            encoder,
            decoders,
            tasks=cfg.tasks,
            task_weights=cfg.task_weights,
        )

        self.module = module
        module.save_hyperparameters(self.cfg)

    def load_datasets(self):
        log.info("Loading NYUv2 dataset")
        data_path = str(Path(self.cfg.data_dir) / "nyuv2")

        self.train_dataset = NYUv2(root=data_path, train=True)
        self.val_dataset = NYUv2(root=data_path, train=False)


@hydra.main(
    config_path=os.path.join(
        importlib.import_module("fusion_bench").__path__[0], "../config"
    ),
    config_name="nyuv2_mtl_train.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    if cfg.print_config:
        rich_print(
            Syntax(
                OmegaConf.to_yaml(cfg),
                "yaml",
                tab_size=2,
                line_numbers=True,
            )
        )
    program = Program(cfg)
    program.run()


if __name__ == "__main__":
    main()
