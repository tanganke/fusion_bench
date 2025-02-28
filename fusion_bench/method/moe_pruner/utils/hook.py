from abc import abstractmethod
from typing import Tuple

from torch import Tensor, nn


class BaseHookFn:
    def __init__(self, module: nn.Module):
        self.module = module

    @abstractmethod
    def compute(self) -> Tensor:
        """
        Compute the importance scores.
        """
        pass

    @abstractmethod
    def __call__(self, router, inps: Tuple[Tensor], out: Tensor):
        """
        Hook function to be called during the forward pass.
        """
        pass
