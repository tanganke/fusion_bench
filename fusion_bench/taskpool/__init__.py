# flake8: noqa F401
import sys

from typing_extensions import TYPE_CHECKING

from fusion_bench.utils.lazy_imports import LazyImporter

_import_structure = {
    "base_pool": ["BaseTaskPool"],
    "clip_vision": [
        "CLIPVisionModelTaskPool",
        "SparseWEMoECLIPVisionModelTaskPool",
        "RankoneMoECLIPVisionModelTaskPool",
    ],
    "dummy": ["DummyTaskPool"],
    "flan_t5": ["FlanT5GLUETextGenerationTaskPool"],
    "gpt2_text_classification": ["GPT2TextClassificationTaskPool"],
    "llama": ["LlamaTestGenerationTaskPool"],
    "nyuv2_taskpool": ["NYUv2TaskPool"],
    "openclip_vision": ["OpenCLIPVisionModelTaskPool"],
}


if TYPE_CHECKING:
    from .base_pool import BaseTaskPool
    from .clip_vision import (
        CLIPVisionModelTaskPool,
        RankoneMoECLIPVisionModelTaskPool,
        SparseWEMoECLIPVisionModelTaskPool,
    )
    from .dummy import DummyTaskPool
    from .flan_t5 import FlanT5GLUETextGenerationTaskPool
    from .gpt2_text_classification import GPT2TextClassificationTaskPool
    from .llama import LlamaTestGenerationTaskPool
    from .nyuv2_taskpool import NYUv2TaskPool
    from .openclip_vision import OpenCLIPVisionModelTaskPool
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
    )
