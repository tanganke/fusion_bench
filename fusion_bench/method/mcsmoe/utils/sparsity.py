import torch

from .constants import FP32_EPS


def compute_weight_stable_rank(weight: torch.Tensor) -> float:
    """
    Compute the stable rank of a weight matrix

    Parameters
    ----------
    weight: torch.Tensor
        The weight matrix

    Returns
    -------
    float
        The stable rank of the weight matrix

    Examples
    --------
    >>> import torch
    >>> compute_weight_stable_rank(torch.randn(100, 100)) > 20
    True
    """
    singular_values = torch.linalg.svdvals(weight)
    return torch.sum(singular_values**2).item() / (
        torch.max(singular_values).item() ** 2 + FP32_EPS
    )
