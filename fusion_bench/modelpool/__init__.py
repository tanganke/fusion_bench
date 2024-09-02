import sys
from typing import TYPE_CHECKING

from fusion_bench.utils.lazy_imports import LazyImporter

_import_structure = {
    "base_pool": ["BaseModelPool"],
    "clip_vision": ["CLIPVisionModelPool"],
    "nyuv2_modelpool": ["NYUv2ModelPool"],
    "huggingface_automodel": ["AutoModelPool"],
    "causal_lm": [
        "AutoModelForCausalLMPool",
        "LLamaForCausalLMPool",
        "MistralForCausalLMPool",
    ],
    "AutoModelForSeq2SeqLM": ["AutoModelForSeq2SeqLMPool"],
    "PeftModelForSeq2SeqLM": ["PeftModelForSeq2SeqLMPool"],
}


if TYPE_CHECKING:
    from .AutoModelForSeq2SeqLM import AutoModelForSeq2SeqLMPool
    from .base_pool import BaseModelPool
    from .causal_lm import (
        AutoModelForCausalLMPool,
        LLamaForCausalLMPool,
        MistralForCausalLMPool,
    )
    from .clip_vision import CLIPVisionModelPool
    from .huggingface_automodel import AutoModelPool
    from .nyuv2_modelpool import NYUv2ModelPool
    from .PeftModelForSeq2SeqLM import PeftModelForSeq2SeqLMPool

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
    )
