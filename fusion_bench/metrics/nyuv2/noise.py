from typing import List, cast

import torch
from torch import Tensor, nn
from torchmetrics import Metric


class NoiseMetric(Metric):
    def __init__(self):
        super().__init__()

    def update(self, preds: Tensor, target: Tensor):
        pass

    def compute(self):
        return [1]
