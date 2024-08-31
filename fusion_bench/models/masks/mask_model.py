from typing import Dict, List, Literal, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fusion_bench.models import ParameterDictModel
from fusion_bench.utils.type import StateDictType


def mask_sparsity(mask: Dict[str, Tensor]):
    total = 0
    non_zero = 0
    for name, m in mask.items():
        total += m.numel()
        non_zero += m.sum().item()
    return non_zero / total


def to_state_dict(
    state_dict_or_model: Union[StateDictType, nn.Module],
    ignore_untrained_params: bool = False,
    ignore_keys: List[str] = [],
    keep_vars: bool = False,
):
    """
    Convert a PyTorch model or state dictionary to a state dictionary, optionally ignoring untrained parameters, specified keys, and keeping differentiable.

    Args:
        state_dict_or_model: Either a PyTorch model (nn.Module) or a state dictionary (StateDictType).
        ignore_untrained: If True, ignore parameters that are not being trained (i.e., those with requires_grad=False).
        ignore_keys: A list of keys to ignore when converting to a state dictionary.
        keep_vars: If True, keeps the Variable wrappers around the tensor data; otherwise, the returned state dictionary will have tensors only.


    Returns:
        A state dictionary (StateDictType) containing the model's parameters.
    """
    if isinstance(state_dict_or_model, nn.Module):
        state_dict: StateDictType = state_dict_or_model.state_dict(keep_vars=True)
        if ignore_untrained_params:
            for name, param in state_dict.items():
                if not param.requires_grad:
                    state_dict.pop(name)
    else:
        state_dict = state_dict_or_model

    for key in ignore_keys:
        state_dict.pop(key, None)

    if not keep_vars:
        for name, param in state_dict.items():
            state_dict[name] = param.detach()
    return state_dict


def get_masked_state_dict(
    state_dict_or_model: Union[StateDictType, nn.Module], mask: Dict[str, Tensor]
):
    state_dict = to_state_dict(state_dict_or_model)
    masked_state_dict = {}
    for name, m in mask.items():
        masked_state_dict[name] = state_dict[name] * m
    return masked_state_dict


class MaskModel(ParameterDictModel):
    def __init__(
        self,
        state_dict_or_model: Union[StateDictType, nn.Module],
        ignore_keys: List[str] = [],
        ignore_untrained_params: bool = True,
        parameter_type: Literal["probs", "logits"] = None,
    ):
        # Convert the model or state dictionary to a state dictionary
        state_dict = to_state_dict(
            state_dict_or_model,
            ignore_untrained_params=ignore_untrained_params,
            ignore_keys=ignore_keys,
        )

        # Convert the tensor dictionary to a parameter dictionary
        parameters = {}
        for name, param in state_dict.items():
            parameters[name] = nn.Parameter(torch.zeros_like(param), requires_grad=True)

        super().__init__(parameters)
        self.parameter_type = parameter_type

    def _param_to_distribution(
        self,
        param: Tensor,
        mask_type: Literal["discrete", "continuous"],
        temperature: float = 0.5,
        **kwargs,
    ):
        if self.parameter_type == "probs":
            if mask_type == "discrete":
                return torch.distributions.Bernoulli(probs=param, **kwargs)
            elif mask_type == "continuous":
                return torch.distributions.RelaxedBernoulli(
                    probs=param, temperature=temperature, **kwargs
                )
        elif self.parameter_type == "logits":
            if mask_type == "discrete":
                return torch.distributions.Bernoulli(logits=param, **kwargs)
            elif mask_type == "continuous":
                return torch.distributions.RelaxedBernoulli(
                    logits=param, temperature=temperature, **kwargs
                )
        raise ValueError(f"Invalid parameter type: {self.parameter_type}")

    def get_distribution(
        self,
        mask_type: Literal["discrete", "continuous"],
        **kwargs,
    ):
        return {
            name: self._param_to_distribution(param, mask_type=mask_type, **kwargs)
            for name, param in self.named_parameters()
        }

    def sample_mask(
        self,
        mask_type: Literal["discrete", "continuous"] = "discrete",
        **kwargs,
    ):
        mask = {}
        for name, param in self.named_parameters():
            dist = self._param_to_distribution(param, mask_type, **kwargs)
            if mask_type == "discrete":
                mask[name] = dist.sample()
            elif mask_type == "continuous":
                mask[name] = dist.rsample()
            else:
                raise ValueError(f"Invalid mask type: {mask_type}")
        return mask

    def fill_(self, value: float):
        """
        Fills the parameters with a given value.

        Args:
            value (float): The value to fill the parameters with.
        """
        for param in self.parameters():
            param.data.fill_(value)

    def clamp_(self, min=None, max=None):
        for param in self.parameters():
            param.data = param.data.clamp(min, max)

    def sparsity(self):
        mask = self.sample_mask(mask_type="discrete")
        total = 0
        non_zero = 0
        for name, m in mask.items():
            total += m.numel()
            non_zero += m.sum().item()
        return non_zero / total
