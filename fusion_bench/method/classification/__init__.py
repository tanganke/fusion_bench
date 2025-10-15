# flake8: noqa F401
import sys
from typing import TYPE_CHECKING

from fusion_bench.utils.lazy_imports import LazyImporter

_import_structure = {
    "clip_finetune": ["ImageClassificationFineTuningForCLIP"],
    "continual_clip_finetune": ["ContinualImageClassificationFineTuningForCLIP"],
    "image_classification_finetune": [
        "ImageClassificationFineTuning",
        "ImageClassificationFineTuning_Test",
    ],
}

if TYPE_CHECKING:
    from .clip_finetune import ImageClassificationFineTuningForCLIP
    from .continual_clip_finetune import ContinualImageClassificationFineTuningForCLIP
    from .image_classification_finetune import (
        ImageClassificationFineTuning,
        ImageClassificationFineTuning_Test,
    )
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
    )
