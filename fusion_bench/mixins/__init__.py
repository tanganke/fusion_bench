# flake8: noqa F401
import sys

from typing_extensions import TYPE_CHECKING

from fusion_bench.utils.lazy_imports import LazyImporter

_import_structure = {
    "clip_classification": ["CLIPClassificationMixin"],
    "fabric_training": ["FabricTrainingMixin"],
    "hydra_config": ["HydraConfigMixin"],
    "lightning_fabric": ["LightningFabricMixin"],
    "openclip_classification": ["OpenCLIPClassificationMixin"],
    "serialization": ["YAMLSerializationMixin", "BaseYAMLSerializableModel"],
    "simple_profiler": ["SimpleProfilerMixin"],
}

if TYPE_CHECKING:
    from .clip_classification import CLIPClassificationMixin
    from .fabric_training import FabricTrainingMixin
    from .hydra_config import HydraConfigMixin
    from .lightning_fabric import LightningFabricMixin
    from .openclip_classification import OpenCLIPClassificationMixin
    from .serialization import BaseYAMLSerializableModel, YAMLSerializationMixin
    from .simple_profiler import SimpleProfilerMixin
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
    )
