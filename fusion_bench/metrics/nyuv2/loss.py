import torch
from torch import Tensor, nn


def segmentation_loss(pred: Tensor, gt: Tensor):
    """
    Compute cross-entropy loss for semantic segmentation.

    Args:
        pred: Predicted segmentation logits of shape (batch_size, num_classes, height, width).
        gt: Ground truth segmentation labels of shape (batch_size, height, width).
            Pixels with value -1 are ignored in the loss computation.

    Returns:
        Tensor: Scalar loss value.
    """
    return nn.functional.cross_entropy(pred, gt.long(), ignore_index=-1)


def depth_loss(pred: Tensor, gt: Tensor):
    """
    Compute L1 loss for depth estimation with binary masking.

    This loss function calculates the absolute error between predicted and ground truth
    depth values, but only for valid pixels (where ground truth depth is non-zero).

    Args:
        pred: Predicted depth values of shape (batch_size, 1, height, width).
        gt: Ground truth depth values of shape (batch_size, 1, height, width).
            Pixels with sum of 0 across channels are considered invalid and masked out.

    Returns:
        Tensor: Scalar loss value averaged over valid pixels.
    """
    binary_mask = (torch.sum(gt, dim=1) != 0).float().unsqueeze(1).to(pred.device)
    loss = torch.sum(torch.abs(pred - gt) * binary_mask) / torch.nonzero(
        binary_mask, as_tuple=False
    ).size(0)
    return loss


def normal_loss(pred: Tensor, gt: Tensor):
    """
    Compute cosine similarity loss for surface normal prediction.

    This loss measures the angular difference between predicted and ground truth
    surface normals using normalized cosine similarity (1 - dot product).

    Args:
        pred: Predicted surface normals of shape (batch_size, 3, height, width).
              Will be L2-normalized before computing loss.
        gt: Ground truth surface normals of shape (batch_size, 3, height, width).
            Already normalized on NYUv2 dataset. Pixels with sum of 0 are invalid.

    Returns:
        Tensor: Scalar loss value (1 - mean cosine similarity) over valid pixels.
    """
    # gt has been normalized on the NYUv2 dataset
    pred = pred / torch.norm(pred, p=2, dim=1, keepdim=True)
    binary_mask = (torch.sum(gt, dim=1) != 0).float().unsqueeze(1).to(pred.device)
    loss = 1 - torch.sum((pred * gt) * binary_mask) / torch.nonzero(
        binary_mask, as_tuple=False
    ).size(0)
    return loss


loss_fn = {
    "segmentation": segmentation_loss,
    "depth": depth_loss,
    "normal": normal_loss,
}
