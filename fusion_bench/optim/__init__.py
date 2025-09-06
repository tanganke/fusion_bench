import sys
from typing import TYPE_CHECKING

from fusion_bench.utils.lazy_imports import LazyImporter

from . import lr_scheduler

_extra_objects = {
    "lr_scheduler": lr_scheduler,
}
_import_structure = {
    "exception": [
        "NoClosureError",
        "NoSparseGradientError",
        "NegativeLRError",
        "NegativeStepError",
        "ZeroParameterSizeError",
    ],
    "mezo": ["MeZO"],
    "muon": ["Muon"],
}

if TYPE_CHECKING:
    from .exception import (
        NegativeLRError,
        NegativeStepError,
        NoClosureError,
        NoSparseGradientError,
        ZeroParameterSizeError,
    )
    from .mezo import MeZO
    from .muon import Muon

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
        extra_objects=_extra_objects,
    )
