import sys
from typing import TYPE_CHECKING

from fusion_bench.utils.lazy_imports import LazyImporter

_import_structure = {
    "linear_warmup": [
        "BaseLinearWarmupScheduler",
        "LinearWarmupScheduler",
        "CosineDecayWithWarmup",
        "PolySchedulerWithWarmup",
    ],
}

if TYPE_CHECKING:
    from .linear_warmup import (
        BaseLinearWarmupScheduler,
        CosineDecayWithWarmup,
        LinearWarmupScheduler,
        PolySchedulerWithWarmup,
    )
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
    )
