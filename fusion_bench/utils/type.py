import sys
from typing import Dict, List, Mapping

from typing_extensions import TypeAlias

try:
    import torch
    from torch import Tensor

    StateDictType: TypeAlias = Dict[str, Tensor]
except ImportError:
    pass

ModuleType = type(sys)
