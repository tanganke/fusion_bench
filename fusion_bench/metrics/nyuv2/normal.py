from typing import List, cast

import numpy as np
import torch
from torch import Tensor, nn
from torchmetrics import Metric


class NormalMetric(Metric):
    """
    Metric for evaluating surface normal prediction on NYUv2 dataset.

    This metric computes angular error statistics between predicted and ground truth
    surface normals, including mean, median, and percentage of predictions within
    specific angular thresholds (11.25°, 22.5°, 30°).

    Attributes:
        metric_names: List of metric names ["mean", "median", "<11.25", "<22.5", "<30"].
        record: List storing angular errors (in degrees) for all pixels across batches.
    """

    metric_names = ["mean", "median", "<11.25", "<22.5", "<30"]

    def __init__(self):
        """Initialize the NormalMetric with state for recording angular errors."""
        super(NormalMetric, self).__init__()

        self.add_state("record", default=[], dist_reduce_fx="cat")

    def update(self, preds, target):
        """
        Update metric state with predictions and targets from a batch.

        Args:
            preds: Predicted surface normals of shape (batch_size, 3, height, width).
                   Will be L2-normalized before computing errors.
            target: Ground truth surface normals of shape (batch_size, 3, height, width).
                   Already normalized on NYUv2 dataset. Pixels with sum of 0 are invalid.
        """
        # gt has been normalized on the NYUv2 dataset
        preds = preds / torch.norm(preds, p=2, dim=1, keepdim=True)
        binary_mask = torch.sum(target, dim=1) != 0
        error = (
            torch.acos(
                torch.clamp(
                    torch.sum(preds * target, 1).masked_select(binary_mask), -1, 1
                )
            )
            .detach()
            .cpu()
            .numpy()
        )
        error = np.degrees(error)
        self.record.append(torch.from_numpy(error))

    def compute(self):
        """
        Compute final metric values from all recorded angular errors.

        Returns:
            List[Tensor]: A list containing five metrics:
                - mean: Mean angular error in degrees.
                - median: Median angular error in degrees.
                - <11.25: Percentage of pixels with error < 11.25°.
                - <22.5: Percentage of pixels with error < 22.5°.
                - <30: Percentage of pixels with error < 30°.

        Note:
            Returns zeros if no data has been recorded.
        """
        if self.record is None:
            return torch.asarray([0.0, 0.0, 0.0, 0.0, 0.0])

        records = torch.concatenate(self.record)
        return [
            torch.mean(records),
            torch.median(records),
            torch.mean((records < 11.25) * 1.0),
            torch.mean((records < 22.5) * 1.0),
            torch.mean((records < 30) * 1.0),
        ]
