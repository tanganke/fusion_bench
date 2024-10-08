from enum import StrEnum
from typing import Callable, Dict, Union

import torch
from torch import nn


class PruningType(StrEnum):
    UNSTRUCTURED = "unstructured"
    SEMISTRUCTURED = "semistructured"
    STRUCTURED = "structured"


def find_linear_layers(module: nn.Module, layers=[nn.Linear], prefix=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        prefix (str): A prefix to add to the layer names.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    res = {}
    for name, submodule in module.named_modules(prefix=prefix):
        if isinstance(submodule, tuple(layers)):
            res[name] = submodule
    return res


def unstructured_magnitude_prune_(
    weight: torch.Tensor,
    metric_function_or_scores: Union[
        Callable[[torch.Tensor], torch.Tensor], torch.Tensor
    ],
    sparsity_ratio: float,
    dtype: torch.dtype = None,
    device: torch.device = None,
    return_pruned_weight: bool = False,
):
    original_device = weight.device
    if callable(metric_function_or_scores):
        W_metric = metric_function_or_scores(weight.to(dtype=dtype, device=device))
    elif isinstance(metric_function_or_scores, torch.Tensor):
        W_metric = metric_function_or_scores.to(dtype=dtype, device=device)
    else:
        raise ValueError(
            "metric_function_or_scores should be either a callable or a tensor"
        )

    W_mask = torch.zeros_like(W_metric) == 1
    sort_res = torch.sort(W_metric, dim=-1, stable=True)
    indices = sort_res[1][:, : int(W_metric.shape[1] * sparsity_ratio)]
    W_mask.scatter_(1, indices, True)
    W_mask = W_mask.to(device=original_device)

    if not return_pruned_weight:
        weight.masked_fill_(W_mask, 0)
        return weight
    else:
        pruned_weight = weight.clone()
        weight.masked_fill_(W_mask, 0)
        pruned_weight.masked_fill_(~W_mask, 0)
        return weight, pruned_weight


def semistructured_magnitude_prune_(
    weight: torch.Tensor,
    metric_function_or_scores: Union[
        Callable[[torch.Tensor], torch.Tensor], torch.Tensor
    ],
    n: int,
    m: int,
    dtype: torch.dtype = None,
    device: torch.device = None,
    return_pruned_weight: bool = False,
):
    original_device = weight.device
    if callable(metric_function_or_scores):
        W_metric = metric_function_or_scores(weight.to(dtype=dtype, device=device))
    elif isinstance(metric_function_or_scores, torch.Tensor):
        W_metric = metric_function_or_scores.to(dtype=dtype, device=device)
    else:
        raise ValueError(
            "metric_function_or_scores should be either a callable or a tensor"
        )
    W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
    for col_idx in range(0, W_metric.shape[1], m):
        tmp = W_metric[:, col_idx : (col_idx + m)].float()
        W_mask.scatter_(
            1,
            col_idx + torch.topk(tmp, n, dim=1, largest=False)[1],
            True,
        )
    W_mask = W_mask.to(device=original_device)
    if not return_pruned_weight:
        weight.masked_fill_(W_mask, 0)
        return weight
    else:
        pruned_weight = weight.clone()
        weight.masked_fill_(W_mask, 0)
        pruned_weight.masked_fill_(~W_mask, 0)
        return weight, pruned_weight


def compute_sparsity(weight: torch.Tensor):
    return (weight == 0).sum() / weight.numel()
