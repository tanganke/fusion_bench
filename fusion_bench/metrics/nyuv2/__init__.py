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
