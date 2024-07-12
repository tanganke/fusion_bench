from .depth import DepthMetric
from .noise import NoiseMetric
from .normal import NormalMetric
from .segmentation import SegmentationMertic

metric_classes = {
    "segmentation": SegmentationMertic,
    "depth": DepthMetric,
    "normal": NormalMetric,
    "noise": NoiseMetric,
}
