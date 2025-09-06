# flake8: noqa F401
import sys
from typing import TYPE_CHECKING

from fusion_bench.utils.lazy_imports import LazyImporter

from . import utils

_extra_objects = {
    "utils": utils,
}
_import_structure = {
    "hf_utils": [
        "create_default_model_card",
        "load_model_card_template",
        "save_pretrained_with_remote_code",
    ],
    "parameter_dict": ["ParameterDictModel"],
    "separate_io": ["separate_load", "separate_save"],
}

if TYPE_CHECKING:
    from .hf_utils import (
        create_default_model_card,
        load_model_card_template,
        save_pretrained_with_remote_code,
    )
    from .parameter_dict import ParameterDictModel
    from .separate_io import separate_load, separate_save
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
        extra_objects=_extra_objects,
    )
