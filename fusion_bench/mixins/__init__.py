# flake8: noqa F401
import sys

from typing_extensions import TYPE_CHECKING

from fusion_bench.utils.lazy_imports import LazyImporter

_import_structure = {
    "lightning_fabric": ["LightningFabricMixin"],
    "serialization": ["YAMLSerializationMixin", "BaseYAMLSerializableModel"],
    "simple_profiler": ["SimpleProfilerMixin"],
    "clip_classification": ["CLIPClassificationMixin"],
    "fabric_training": ["FabricTrainingMixin"],
}

if TYPE_CHECKING:
    from .clip_classification import CLIPClassificationMixin
    from .fabric_training import FabricTrainingMixin
    from .lightning_fabric import LightningFabricMixin
    from .serialization import BaseYAMLSerializableModel, YAMLSerializationMixin
    from .simple_profiler import SimpleProfilerMixin

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
    )
