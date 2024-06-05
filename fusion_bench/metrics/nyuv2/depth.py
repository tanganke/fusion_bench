from typing import List, cast

import numpy as np
import torch
from torch import Tensor, nn
from torchmetrics import Metric


class DepthMetric(Metric):
    metric_names = ["abs_err", "rel_err"]

    def __init__(self):
        super().__init__()

        self.add_state("abs_record", default=[], dist_reduce_fx="cat")
        self.add_state("rel_record", default=[], dist_reduce_fx="cat")
        self.add_state("batch_size", default=[], dist_reduce_fx="cat")

    def reset(self):
        self.abs_record = []
        self.rel_record = []
        self.batch_size = []

    def update(self, preds: Tensor, target: Tensor):
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
        records = torch.stack(
            [torch.stack(self.abs_record), torch.stack(self.rel_record)]
        )
        batch_size = torch.stack(self.batch_size)
        return [(records[i] * batch_size).sum() / batch_size.sum() for i in range(2)]
