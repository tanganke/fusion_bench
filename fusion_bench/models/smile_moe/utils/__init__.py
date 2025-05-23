from typing import List

import torch
from torch import Tensor

from .svd_utils import svd

__all__ = ["svd_utils", "_is_all_zeros"]


def _is_all_zeros(tensor: Tensor | List[Tensor]) -> bool:
    """
    Check if a tensor or a list of tensors are all zeros.

    Args:
        tensor (Tensor | List[Tensor]): A tensor or a list of tensors.

    Returns:
        bool: True if all elements are zeros, False otherwise.
    """
    if isinstance(tensor, Tensor):
        return torch.allclose(tensor, torch.zeros_like(tensor))
    else:
        return all(_is_all_zeros(t) for t in tensor)
