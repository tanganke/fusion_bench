from typing import Callable, Dict, Union

import torch


def unstructured_magnitude_prune_(
    weight: torch.Tensor,
    metric_function: Callable[[torch.Tensor], torch.Tensor],
    sparsity_ratio: float,
    dtype: torch.dtype = None,
    device: torch.device = None,
):
    original_device = weight.device
    W_metric = metric_function(weight.to(dtype=dtype, device=device))
    thresh = torch.sort(W_metric.flatten())[0][int(weight.numel() * sparsity_ratio)]
    W_mask = W_metric <= thresh

    weight[W_mask.to(device=original_device)] = 0
    return weight


def semistructured_magnitude_prune_(
    weight: torch.Tensor,
    metric_function: Callable[[torch.Tensor], torch.Tensor],
    n: int,
    m: int,
    dtype: torch.dtype = None,
    device: torch.device = None,
):
    original_device = weight.device
    W_metric = metric_function(weight.to(dtype=dtype, device=device))
    W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
    for col_idx in range(0, W_metric.shape[1], m):
        tmp = W_metric[:, col_idx : (col_idx + m)].float()
        W_mask.scatter_(
            1,
            col_idx + torch.topk(tmp, n, dim=1, largest=False)[1],
            True,
        )

    weight[W_mask.to(device=original_device)] = 0
    return weight


def compute_sparsity(weight: torch.Tensor):
    return (weight == 0).sum() / weight.numel()
