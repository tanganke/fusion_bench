# flake8: noqa F401
import sys
from typing import Dict, List, Mapping, TypeVar

from typing_extensions import TypeAlias

try:
    import torch
    from torch import Tensor

    StateDictType: TypeAlias = Dict[str, Tensor]
except ImportError:
    pass


ModuleType = type(sys)
T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")

__all__ = ["StateDictType", "ModuleType", "T", "T1", "T2", "T3", "T4"]
