from typing import List, cast

import torch
from torch import Tensor, nn
from torchmetrics import Metric


class SegmentationMetric(Metric):
    """
    Metric for evaluating semantic segmentation on NYUv2 dataset.

    This metric computes mean Intersection over Union (mIoU) and pixel accuracy
    for multi-class segmentation tasks.

    Attributes:
        metric_names: List of metric names ["mIoU", "pixAcc"].
        num_classes: Number of segmentation classes (default: 13 for NYUv2).
        record: Confusion matrix of shape (num_classes, num_classes) tracking
                predictions vs ground truth.
    """

    metric_names = ["mIoU", "pixAcc"]

    def __init__(self, num_classes=13):
        """
        Initialize the SegmentationMetric.

        Args:
            num_classes: Number of segmentation classes. Default is 13 for NYUv2 dataset.
        """
        super().__init__()

        self.num_classes = num_classes
        self.add_state(
            "record",
            default=torch.zeros(
                (self.num_classes, self.num_classes), dtype=torch.int64
            ),
            dist_reduce_fx="sum",
        )

    def reset(self):
        """Reset the confusion matrix to zeros."""
        self.record.zero_()

    def update(self, preds: Tensor, target: Tensor):
        """
        Update the confusion matrix with predictions and targets from a batch.

        Args:
            preds: Predicted segmentation logits of shape (batch_size, num_classes, height, width).
                   Will be converted to class predictions via softmax and argmax.
            target: Ground truth segmentation labels of shape (batch_size, height, width).
                   Pixels with negative values or values >= num_classes are ignored.
        """
        preds = preds.softmax(1).argmax(1).flatten()
        target = target.long().flatten()

        k = (target >= 0) & (target < self.num_classes)
        inds = self.num_classes * target[k].to(torch.int64) + preds[k]
        self.record += torch.bincount(inds, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self):
        """
        Compute mIoU and pixel accuracy from the confusion matrix.

        Returns:
            List[Tensor]: A list containing [mIoU, pixel_accuracy]:
                - mIoU: Mean Intersection over Union across all classes.
                - pixel_accuracy: Overall pixel classification accuracy.
        """
        h = cast(Tensor, self.record).float()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        acc = torch.diag(h).sum() / h.sum()
        return [torch.mean(iu), acc]
