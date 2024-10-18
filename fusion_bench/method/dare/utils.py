import torch
from torch import Tensor, nn

from fusion_bench.utils.type import StateDictType


def param_random_drop_(param: Tensor, sparsity_level: float, rescale: bool):
    mask = torch.rand_like(param) > sparsity_level
    param.data = param.data * mask
    if rescale:
        param.data = param.data / (1 - sparsity_level)


def module_random_drop_(tv: nn.Module, sparsity_level: float, rescale: bool):
    for param in tv.parameters():
        param_random_drop_(param, sparsity_level, rescale)


def trainable_state_dict(module: nn.Module):
    return {
        name: param for name, param in module.named_parameters() if param.requires_grad
    }


def module_sub_(a: nn.Module, b: nn.Module):
    for (a_name, a_param), (b_name, b_param) in zip(
        a.named_parameters(), b.named_parameters()
    ):
        assert a_name == b_name, "Mismatch in parameter names"
        a_param.data = a_param.data - b_param.data
    return a
