import sys

from omegaconf import DictConfig
from typing_extensions import TYPE_CHECKING

from fusion_bench.utils.lazy_imports import LazyImporter

_import_structure = {
    "base_pool": ["BaseTaskPool"],
    "clip_vision": ["CLIPVisionModelTaskPool"],
    "dummy": ["DummyTaskPool"],
    "gpt2_text_classification": ["GPT2TextClassificationTaskPool"],
    "flan_t5_glue_text_generation": ["FlanT5GLUETextGenerationTaskPool"],
    "nyuv2_taskpool": ["NYUv2TaskPool"],
}


if TYPE_CHECKING:
    from .base_pool import BaseTaskPool
    from .clip_vision import CLIPVisionModelTaskPool
    from .dummy import DummyTaskPool
    from .flan_t5_glue_text_generation import FlanT5GLUETextGenerationTaskPool
    from .gpt2_text_classification import GPT2TextClassificationTaskPool
    from .nyuv2_taskpool import NYUv2TaskPool

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
    )
