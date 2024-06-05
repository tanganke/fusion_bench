import torch
from torch import Tensor, nn


def segmentation_loss(pred: Tensor, gt: Tensor):
    return nn.functional.cross_entropy(pred, gt.long(), ignore_index=-1)


def depth_loss(pred: Tensor, gt: Tensor):
    binary_mask = (torch.sum(gt, dim=1) != 0).float().unsqueeze(1).to(pred.device)
    loss = torch.sum(torch.abs(pred - gt) * binary_mask) / torch.nonzero(
        binary_mask, as_tuple=False
    ).size(0)
    return loss


def normal_loss(pred: Tensor, gt: Tensor):
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
