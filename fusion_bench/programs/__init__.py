# flake8: noqa F401
import sys
from typing import TYPE_CHECKING

from fusion_bench.utils.lazy_imports import LazyImporter

_import_structure = {
    "base_program": ["BaseHydraProgram"],
    "fabric_fusion_program": ["FabricModelFusionProgram"],
}

if TYPE_CHECKING:
    from .base_program import BaseHydraProgram
    from .fabric_fusion_program import FabricModelFusionProgram
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
    )
