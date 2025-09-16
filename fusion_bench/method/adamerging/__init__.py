# flake8: noqa F401
import sys
from typing import TYPE_CHECKING

from fusion_bench.utils.lazy_imports import LazyImporter

_import_structure = {
    "clip_layer_wise_adamerging": ["CLIPLayerWiseAdaMergingAlgorithm"],
    "clip_task_wise_adamerging": ["CLIPTaskWiseAdaMergingAlgorithm"],
    "flan_t5_layer_wise_adamerging": ["FlanT5LayerWiseAdaMergingAlgorithm"],
    "gpt2_layer_wise_adamerging": ["GPT2LayerWiseAdaMergingAlgorithm"],
    "llama_adamerging": ["LayerWiseAdaMergingForLlamaSFT"],
    "resnet_adamerging": ["ResNetLayerWiseAdamerging", "ResNetTaskWiseAdamerging"],
}

if TYPE_CHECKING:
    from .clip_layer_wise_adamerging import CLIPLayerWiseAdaMergingAlgorithm
    from .clip_task_wise_adamerging import CLIPTaskWiseAdaMergingAlgorithm
    from .flan_t5_layer_wise_adamerging import FlanT5LayerWiseAdaMergingAlgorithm
    from .gpt2_layer_wise_adamerging import GPT2LayerWiseAdaMergingAlgorithm
    from .llama_adamerging import LayerWiseAdaMergingForLlamaSFT
    from .resnet_adamerging import ResNetLayerWiseAdamerging, ResNetTaskWiseAdamerging

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
    )
