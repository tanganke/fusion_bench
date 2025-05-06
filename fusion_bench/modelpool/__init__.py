# flake8: noqa F401
import sys
from typing import TYPE_CHECKING

from fusion_bench.utils.lazy_imports import LazyImporter

_import_structure = {
    "base_pool": ["BaseModelPool"],
    "causal_lm": ["CausalLMPool", "CausalLMBackbonePool"],
    "clip_vision": ["CLIPVisionModelPool"],
    "nyuv2_modelpool": ["NYUv2ModelPool"],
    "huggingface_automodel": ["AutoModelPool"],
    "seq2seq_lm": ["Seq2SeqLMPool"],
    "PeftModelForSeq2SeqLM": ["PeftModelForSeq2SeqLMPool"],
    "openclip_vision": ["OpenCLIPVisionModelPool"],
    "huggingface_gpt2_classification": [
        "HuggingFaceGPT2ClassificationPool",
        "GPT2ForSequenceClassificationPool",
    ],
    "seq_classification_lm": ["SeqenceClassificationModelPool"],
}


if TYPE_CHECKING:
    from .base_pool import BaseModelPool
    from .causal_lm import CausalLMBackbonePool, CausalLMPool
    from .clip_vision import CLIPVisionModelPool
    from .huggingface_automodel import AutoModelPool
    from .huggingface_gpt2_classification import (
        GPT2ForSequenceClassificationPool,
        HuggingFaceGPT2ClassificationPool,
    )
    from .nyuv2_modelpool import NYUv2ModelPool
    from .openclip_vision import OpenCLIPVisionModelPool
    from .PeftModelForSeq2SeqLM import PeftModelForSeq2SeqLMPool
    from .seq2seq_lm import Seq2SeqLMPool
    from .seq_classification_lm import SeqenceClassificationModelPool

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
    )
