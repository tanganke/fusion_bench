# flake8: noqa F401
import sys

from typing_extensions import TYPE_CHECKING

from fusion_bench.utils.lazy_imports import LazyImporter

_import_structure = {
    "base_pool": ["BaseTaskPool"],
    "clip_vision": [
        "CLIPVisionModelTaskPool",
        "SparseWEMoECLIPVisionModelTaskPool",
        "RankoneWEMoECLIPVisionModelTaskPool",
    ],
    "dummy": ["DummyTaskPool"],
    "gpt2_text_classification": ["GPT2TextClassificationTaskPool"],
    "nyuv2_taskpool": ["NYUv2TaskPool"],
    "llama": ["LlamaTestGenerationTaskPool"],
}


if TYPE_CHECKING:
    from .base_pool import BaseTaskPool
    from .clip_vision import (
        CLIPVisionModelTaskPool,
        RankoneWEMoECLIPVisionModelTaskPool,
        SparseWEMoECLIPVisionModelTaskPool,
    )
    from .dummy import DummyTaskPool
    from .gpt2_text_classification import GPT2TextClassificationTaskPool
    from .llama import LlamaTestGenerationTaskPool
    from .nyuv2_taskpool import NYUv2TaskPool

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
    )
