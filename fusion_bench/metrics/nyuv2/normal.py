from typing import List, cast

import numpy as np
import torch
from torch import Tensor, nn
from torchmetrics import Metric


class NormalMetric(Metric):
    metric_names = ["mean", "median", "<11.25", "<22.5", "<30"]

    def __init__(self):
        super(NormalMetric, self).__init__()

        self.add_state("record", default=[], dist_reduce_fx="cat")

    def update(self, preds, target):
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
        returns mean, median, and percentage of pixels with error less than 11.25, 22.5, and 30 degrees ("mean", "median", "<11.25", "<22.5", "<30")
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
