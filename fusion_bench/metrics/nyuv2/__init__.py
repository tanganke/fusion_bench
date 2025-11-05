"""
NYUv2 Dataset Metrics Module.

This module provides metric classes and loss functions for evaluating multi-task learning
models on the NYUv2 dataset. NYUv2 is a popular indoor scene understanding dataset that
includes multiple tasks: semantic segmentation, depth estimation, and surface normal prediction.

Available Metrics:
    - SegmentationMetric: Computes mIoU and pixel accuracy for semantic segmentation.
    - DepthMetric: Computes absolute and relative errors for depth estimation.
    - NormalMetric: Computes angular errors for surface normal prediction.
    - NoiseMetric: Placeholder metric for noise evaluation.

Usage:
    ```python
    from fusion_bench.metrics.nyuv2 import SegmentationMetric, DepthMetric

    # Initialize metrics
    seg_metric = SegmentationMetric(num_classes=13)
    depth_metric = DepthMetric()

    # Update with predictions and targets
    seg_metric.update(seg_preds, seg_targets)
    depth_metric.update(depth_preds, depth_targets)

    # Compute final metrics
    miou, pix_acc = seg_metric.compute()
    abs_err, rel_err = depth_metric.compute()
    ```
"""

from .depth import DepthMetric
from .noise import NoiseMetric
from .normal import NormalMetric
from .segmentation import SegmentationMetric

metric_classes = {
    "segmentation": SegmentationMetric,
    "depth": DepthMetric,
    "normal": NormalMetric,
    "noise": NoiseMetric,
}
