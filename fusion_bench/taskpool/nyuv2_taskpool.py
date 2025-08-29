import logging
from pathlib import Path

import lightning as L
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from fusion_bench.compat.taskpool.base_pool import TaskPool
from fusion_bench.dataset.nyuv2 import NYUv2
from fusion_bench.models.nyuv2.lightning_module import NYUv2MTLModule
from fusion_bench.models.nyuv2.resnet_dilated import ResnetDilated

log = logging.getLogger(__name__)


class NYUv2TaskPool(TaskPool):
    """Task pool for multi-task learning evaluation on the NYUv2 dataset.

    This task pool provides evaluation capabilities for multi-task learning models
    on the NYU Depth V2 (NYUv2) dataset, which is a popular benchmark for indoor
    scene understanding. The dataset supports multiple computer vision tasks
    including semantic segmentation, depth estimation, and surface normal prediction.

    The task pool is designed to work with encoder-decoder architectures where
    a shared encoder processes input images and task-specific decoders generate
    predictions for different tasks. It integrates with PyTorch Lightning for
    streamlined training and evaluation workflows.

    Supported Tasks:
        - Semantic segmentation
        - Depth estimation
        - Surface normal prediction
    """

    _trainer: L.Trainer = None

    def __init__(self, taskpool_config: DictConfig):
        """Initialize the NYUv2 task pool with configuration settings.

        Args:
            taskpool_config: Configuration object containing all necessary
                parameters for the task pool, including:
                - data_dir: Path to the directory containing NYUv2 dataset
                - tasks: List of tasks to evaluate (e.g., ["semantic", "depth"])
                - batch_size: Batch size for evaluation data loader
                - num_workers: Number of worker processes for data loading
        """
        self.config = taskpool_config

    def load_datasets(self):
        log.info("Loading NYUv2 dataset")
        data_path = str(Path(self.config.data_dir) / "nyuv2")

        train_dataset = NYUv2(root=data_path, train=True)
        val_dataset = NYUv2(root=data_path, train=False)
        return train_dataset, val_dataset

    @property
    def trainer(self):
        if self._trainer is None:
            self._trainer = L.Trainer(devices=1)
        return self._trainer

    def get_decoders(self):
        from fusion_bench.modelpool.nyuv2_modelpool import NYUv2ModelPool

        modelpool: NYUv2ModelPool = self._program.modelpool
        decoders = nn.ModuleDict()
        for task in self.config.tasks:
            decoders[task] = modelpool.load_model(task, encoder_only=False).decoders[
                task
            ]
        return decoders

    def evaluate(self, encoder: ResnetDilated):
        model = NYUv2MTLModule(
            encoder,
            self.get_decoders(),
            tasks=self.config.tasks,
            task_weights=[1] * len(self.config.tasks),
        )
        _, val_dataset = self.load_datasets()
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        report = self.trainer.validate(model, val_loader)
        if isinstance(report, list) and len(report) == 1:
            report = report[0]
        return report
