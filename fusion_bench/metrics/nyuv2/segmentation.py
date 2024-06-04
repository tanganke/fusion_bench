from typing import List, cast

import torch
from torch import Tensor, nn
from torchmetrics import Metric


class SegmentationMertic(Metric):
    metric_names = ["mIoU", "pixAcc"]

    def __init__(self, num_classes=13):
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
        self.record.zero_()

    def update(self, preds: Tensor, target: Tensor):
        preds = preds.softmax(1).argmax(1).flatten()
        target = target.long().flatten()

        k = (target >= 0) & (target < self.num_classes)
        inds = self.num_classes * target[k].to(torch.int64) + preds[k]
        self.record += torch.bincount(inds, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self):
        """
        return mIoU and pixel accuracy
        """
        h = cast(Tensor, self.record).float()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        acc = torch.diag(h).sum() / h.sum()
        return [torch.mean(iu), acc]
