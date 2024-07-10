import torch
from torch import Tensor


def entropy_loss(logits: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute the entropy loss of a set of logits.

    Args:
        logits (Tensor): The logits to compute the entropy loss of.

    Returns:
        Tensor: The entropy loss of the logits.
    """
    assert (
        logits.dim() == 2
    ), f"Expected logits to have 2 dimensions, found {logits.dim()}, {logits.size()=}"
    probs = torch.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + eps), dim=-1).mean()
