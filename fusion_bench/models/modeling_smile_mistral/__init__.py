# flake8: noqa F401
from typing import TYPE_CHECKING

from transformers.utils.import_utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

_import_structure = {
    "configuration_smile_mistral": ["SmileMistralConfig"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_smile_mistral"] = [
        "SmileMistralForCausalLM",
        "SmileMistralModel",
        "SmileMistralPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_smile_mistral import SmileMistralConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_smile_mistral import (
            SmileMistralForCausalLM,
            SmileMistralModel,
            SmileMistralPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
