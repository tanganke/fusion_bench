from typing import List, cast

import torch
from torch import Tensor, nn
from torchmetrics import Metric


class NoiseMetric(Metric):
    """
    A placeholder metric for noise evaluation on NYUv2 dataset.

    This metric currently serves as a placeholder and always returns a value of 1.
    It can be extended in the future to include actual noise-related metrics.

    Note:
        This is a dummy implementation that doesn't perform actual noise measurements.
    """

    def __init__(self):
        """Initialize the NoiseMetric."""
        super().__init__()

    def update(self, preds: Tensor, target: Tensor):
        """
        Update metric state (currently a no-op).

        Args:
            preds: Predicted values (unused).
            target: Ground truth values (unused).
        """
        pass

    def compute(self):
        """
        Compute the metric value.

        Returns:
            List[int]: A list containing [1] as a placeholder value.
        """
        return [1]
