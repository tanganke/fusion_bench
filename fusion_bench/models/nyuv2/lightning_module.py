import itertools
from typing import Dict, List, Literal, Optional, cast

import lightning as L
import torch.nn.functional as F
from torch import Tensor, nn
from torchmetrics import Metric

from fusion_bench.metrics.nyuv2 import metric_classes
from fusion_bench.metrics.nyuv2.loss import loss_fn


class NYUv2Model(nn.Module):
    image_size = (288, 384)

    def __init__(
        self,
        encoder: nn.Module,
        decoders: nn.ModuleDict,
    ):
        R"""
        Args:
            encoder: The encoder module.
            decoders: A dictionary of the decoder modules.
        """
        super().__init__()

        self.encoder = encoder
        self.decoders = decoders

    def encode(self, images: Tensor) -> Tensor:
        return self.encoder(images)

    def decode(self, features: Tensor, key: str) -> Tensor:
        return self.decoders[key](features)

    def model_parameters(self):
        "parameters of the encoder and the decoders"
        return itertools.chain(
            self.encoder.parameters(),
            self.decoders.parameters(),
        )

    def _compute_loss(
        self,
        outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        task: Literal["segmentation", "depth", "normal"] | List[str],
    ) -> Tensor | List[Tensor]:
        if isinstance(task, str):
            return loss_fn[task](outputs[task], targets[task])
        else:
            return [loss_fn[t](outputs[t], targets[t]) for t in task]

    def forward(
        self, images: Tensor, tasks: Optional[List[str]] = None
    ) -> Dict[str, Tensor]:
        features = self.encode(images)

        if tasks is None:
            tasks = self.decoders.keys()
        outputs = {}
        for task in tasks:
            outputs[task] = F.interpolate(
                self.decode(features, task),
                self.image_size,
                mode="bilinear",
                align_corners=True,
            )

        return outputs


class NYUv2MTLModule(NYUv2Model, L.LightningModule):
    R"""
    multi-task learning module for NYUv2 dataset, with weighted loss
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoders: nn.ModuleDict,
        tasks: List[str],
        task_weights: List[float],
    ):
        R"""
        By using this module, you optimize the weighted sum of the losses of the tasks.
        The joint loss is defined as:

        .. math::
            \mathcal{L} = \sum_{i=1}^{N} w_i \mathcal{L}_i

        where :math:`N` is the number of tasks, :math:`w_i` is the weight for the task :math:`i`,

        Args:
            encoder (nn.Module): The encoder module.
            decoders (nn.ModuleDict): A dictionary of the decoder modules.
            tasks (List[str]): A list of tasks.
            task_weights (List[float]): A list of weights for each task.

        Raises:
            AssertionError: If tasks and task_weights do not have the same length.
        """
        assert len(tasks) == len(
            task_weights
        ), "tasks and task_weights must have the same length"

        super().__init__(encoder, decoders)
        self.tasks = tasks
        self.task_weights = task_weights

        self.metrics = nn.ModuleDict({t: metric_classes[t]() for t in self.tasks})

    # training

    def on_train_epoch_start(self):
        for t in self.tasks:
            self.metrics[t].reset()

    def _single_step(self, images, targets):
        outputs = self(images, self.tasks)
        losses = self._compute_loss(outputs, targets, self.tasks)
        weighted_loss = sum([w * l for w, l in zip(self.task_weights, losses)])

        for t in self.metrics:
            self.metrics[t].update(outputs[t], targets[t])

        return {
            "outputs": outputs,
            "losses": losses,
            "weighted_loss": weighted_loss,
        }

    def training_step(self, batch, batch_idx: int):
        images, targets = batch
        results = self._single_step(images, targets)

        outputs = results["outputs"]
        losses = results["losses"]
        weighted_loss = results["weighted_loss"]

        for i, t in enumerate(self.tasks):
            self.log(f"train/{t}_loss", losses[i])
        self.log("train/loss", weighted_loss, prog_bar=True)

        return weighted_loss

    def on_train_epoch_end(self) -> None:
        for t in self.tasks:
            metrics = cast(Metric, self.metrics[t]).compute()
            for metric_name, metric_value in zip(self.metrics[t].metric_names, metrics):
                self.log(f"train/{t}_{metric_name}", metric_value)

    # validation

    def on_validation_epoch_start(self) -> None:
        for t in self.tasks:
            self.metrics[t].reset()

    def validation_step(self, batch, batch_idx: int):
        images, targets = batch
        results = self._single_step(images, targets)

        outputs = results["outputs"]
        losses = results["losses"]
        weighted_loss = results["weighted_loss"]

        for i, t in enumerate(self.tasks):
            self.log(f"val/{t}_loss", losses[i])
        self.log("val/loss", weighted_loss, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        for t in self.tasks:
            metrics = cast(Metric, self.metrics[t]).compute()
            for metric_name, metric_value in zip(self.metrics[t].metric_names, metrics):
                self.log(f"val/{t}_{metric_name}", metric_value)
