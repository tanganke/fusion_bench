from tqdm.autonotebook import tqdm
from .base_task import BaseTask
from torch import nn, Tensor
from torch.nn import functional as F
from torchmetrics import Accuracy, MeanMetric
from torchmetrics.classification.accuracy import MulticlassAccuracy
from abc import abstractmethod


class ImageClassificationTask(BaseTask):
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

    def evaluate(self, classifier: nn.Module):
        self.accuracy: MulticlassAccuracy = Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.loss_metric = MeanMetric()

        for batch in (
            pbar := tqdm(
                self.test_loader, desc="Evaluating", leave=False, dynamic_ncols=True
            )
        ):
            inputs, targets = batch
            logits: Tensor = classifier(inputs)

            loss = F.cross_entropy(logits, targets)
            self.loss_metric.update(loss)
            acc = self.accuracy(logits, targets)
            pbar.set_postfix({"accuracy": acc.item(), "loss": loss.item()})

        acc = self.accuracy.compute().item()
        loss = self.loss_metric.compute().item()
        return {"accuracy": acc, "loss": loss}
