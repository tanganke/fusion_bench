from typing import Dict, List, Mapping

from torch import Tensor
from typing_extensions import TypeAlias

_StateDict: TypeAlias = Mapping[str, Tensor]
