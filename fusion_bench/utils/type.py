# flake8: noqa F401
import sys
from typing import Dict, List, Mapping, TypeVar

from typing_extensions import TypeAlias

try:
    import torch
    from torch import Tensor, nn

    StateDictType: TypeAlias = Dict[str, Tensor]
    BoolStateDictType: TypeAlias = Dict[str, torch.BoolTensor]
    TorchModelType = TypeVar("TorchModelType", bound=nn.Module)

except ImportError:
    pass


PyModuleType = type(sys)
T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")

__all__ = [
    "StateDictType",
    "PyModuleType",
    "TorchModelType",
    "T",
    "T1",
    "T2",
    "T3",
    "T4",
]
