from typing import List, cast

import numpy as np
import torch
from torch import Tensor, nn
from torchmetrics import Metric


class DepthMetric(Metric):
    """
    Metric for evaluating depth estimation performance on NYUv2 dataset.

    This metric computes absolute error and relative error for depth predictions,
    properly handling the binary mask to exclude invalid depth regions.

    Attributes:
        metric_names: List of metric names ["abs_err", "rel_err"].
        abs_record: List storing absolute error values for each batch.
        rel_record: List storing relative error values for each batch.
        batch_size: List storing batch sizes for weighted averaging.
    """

    metric_names = ["abs_err", "rel_err"]

    def __init__(self):
        """Initialize the DepthMetric with state variables for tracking errors."""
        super().__init__()

        self.add_state("abs_record", default=[], dist_reduce_fx="cat")
        self.add_state("rel_record", default=[], dist_reduce_fx="cat")
        self.add_state("batch_size", default=[], dist_reduce_fx="cat")

    def reset(self):
        """Reset all metric states to empty lists."""
        self.abs_record = []
        self.rel_record = []
        self.batch_size = []

    def update(self, preds: Tensor, target: Tensor):
        """
        Update metric states with predictions and targets from a batch.

        Args:
            preds: Predicted depth values of shape (batch_size, 1, height, width).
            target: Ground truth depth values of shape (batch_size, 1, height, width).
                   Pixels with sum of 0 are considered invalid and masked out.
        """
        binary_mask = (torch.sum(target, dim=1) != 0).unsqueeze(1)
        preds = preds.masked_select(binary_mask)
        target = target.masked_select(binary_mask)
        abs_err = torch.abs(preds - target)
        rel_err = torch.abs(preds - target) / target
        abs_err = torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(
            0
        )
        rel_err = torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(
            0
        )
        self.abs_record.append(abs_err)
        self.rel_record.append(rel_err)
        self.batch_size.append(torch.asarray(preds.size(0), device=preds.device))

    def compute(self):
        """
        Compute the final metric values across all batches.

        Returns:
            List[Tensor]: A list containing [absolute_error, relative_error],
                         where each value is the weighted average across all batches.
        """
        records = torch.stack(
            [torch.stack(self.abs_record), torch.stack(self.rel_record)]
        )
        batch_size = torch.stack(self.batch_size)
        return [(records[i] * batch_size).sum() / batch_size.sum() for i in range(2)]
