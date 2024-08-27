import functools
import itertools
import logging
from abc import abstractmethod

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics import Accuracy, MeanMetric
from torchmetrics.classification.accuracy import MulticlassAccuracy
from tqdm.autonotebook import tqdm

from .base_task import BaseTask

log = logging.getLogger(__name__)


class ClassificationTask(BaseTask):
    def __init__(self, task_config):
        super().__init__(task_config)

    @property
    @abstractmethod
    def num_classes(self):
        """
        Returns the number of classes in the dataset.
        """
        pass

    @property
    @abstractmethod
    def test_loader(self):
        """
        Returns a test data loader.
        """
        pass

    @torch.no_grad()
    def evaluate(self, classifier: nn.Module, device=None):
        accuracy: MulticlassAccuracy = Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        classifier.eval()
        loss_metric = MeanMetric()
        # if fast_dev_run is set, we only evaluate on a batch of the data
        if self.config.get("fast_dev_run", False):
            log.info("Running under fast_dev_run mode, evaluating on a single batch.")
            test_loader = itertools.islice(self.test_loader, 1)
        else:
            test_loader = self.test_loader

        for batch in (
            pbar := tqdm(
                test_loader, desc="Evaluating", leave=False, dynamic_ncols=True
            )
        ):
            inputs, targets = batch
            if device is not None:
                inputs, targets = inputs.to(device), targets.to(device)
            logits: Tensor = classifier(inputs)

            loss = F.cross_entropy(logits, targets)
            loss_metric.update(loss.detach().cpu())
            acc = accuracy(logits.detach().cpu(), targets.detach().cpu())
            pbar.set_postfix(
                {
                    "accuracy": accuracy.compute().item(),
                    "loss": loss_metric.compute().item(),
                }
            )

        acc = accuracy.compute().item()
        loss = loss_metric.compute().item()
        results = {"accuracy": acc, "loss": loss}
        return results
