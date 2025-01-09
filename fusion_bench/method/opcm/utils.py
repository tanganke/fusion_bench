from typing import Tuple

import torch
from torch import Tensor, nn

from fusion_bench.utils.parameters import state_dict_to_vector
from fusion_bench.utils.state_dict_arithmetic import state_dict_sub


def _svd(w: Tensor, full_matrices=True) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform Singular Value Decomposition (SVD) on a tensor.

    Args:
        w (Tensor): The input tensor.
        full_matrices (bool): Whether to compute the full-sized U and V matrices.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The U, S, and V matrices from SVD.
    """
    u, s, vh = torch.linalg.svd(
        w, full_matrices=full_matrices, driver="gesvd" if w.is_cuda else None
    )
    v = vh.T
    return u, s, v


def svd(
    w: Tensor, full_matrices=True, accelerator=None
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform SVD on a tensor, optionally using a specified accelerator.

    Args:
        w (Tensor): The input tensor.
        full_matrices (bool): Whether to compute the full-sized U and V matrices.
        accelerator (str): The device to perform the computation on.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The U, S, and V matrices from SVD.
    """
    if accelerator is None:
        return _svd(w, full_matrices=full_matrices)
    original_device = w.device
    w = w.to(accelerator)
    u, s, v = _svd(w)
    return u.to(original_device), s.to(original_device), v.to(original_device)


def frobenius_inner_product(w1: Tensor, w2: Tensor) -> Tensor:
    return torch.trace(w1.T @ w2)


def is_leaf_module(module: nn.Module) -> bool:
    return len(list(module.children())) == 0


def get_task_vector_norm(model: nn.Module, pretrained_model: nn.Module) -> Tensor:
    """
    Get the vector norm of the task model.

    Args:
        model (nn.Module): The task model.
        pretrained_model (nn.Module): The pretrained model.

    Returns:
        Tensor: The vector norm of the task model.
    """
    return torch.linalg.norm(
        state_dict_to_vector(
            state_dict_sub(model.state_dict(), pretrained_model.state_dict())
        )
    )
