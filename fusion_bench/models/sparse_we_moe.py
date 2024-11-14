import functools
import logging
from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
import torch.func
from torch import Tensor, nn
from torch.func import functional_call
from torch.nn import functional as F
from tqdm.auto import tqdm

from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_sub,
    state_dict_weighted_sum,
)
from fusion_bench.utils.type import StateDictType

log = logging.getLogger(__name__)


def join_list(list_of_list: List[List]):
    ans = []
    for l in list_of_list:
        ans.extend(l)
    return ans


def del_attr(obj, names: List[str]):
    """
    Deletes an attribute from an object recursively.

    Args:
        obj (object): Object to delete attribute from.
        names (list): List of attribute names to delete recursively.
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names: List[str], val):
    """
    Sets an attribute of an object recursively.

    Args:
        obj (object): Object to set attribute of.
        names (list): List of attribute names to set recursively.
        val (object): Value to set the attribute to.
    """
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def get_attr(obj, names: List[str]):
    """
    Gets an attribute of an object recursively.

    Args:
        obj (object): Object to get attribute of.
        names (list): List of attribute names to get recursively.

    Returns:
        object: The attribute of the object.
    """
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])


class Depth_0_Gate(nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts), requires_grad=True)

    def init_weight(self, init_lambda: float):
        nn.init.constant_(self.weight, init_lambda)

    def forward(self, *args, **kwargs) -> Tensor:
        return self.weight


class Depth_1_Gate(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, num_experts, bias=True)

    def init_weight(self, init_lambda: float):
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, init_lambda)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.fc(hidden_states)


class Depth_2_Gate(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, num_experts, bias=True)

    def init_weight(self, init_lambda: float):
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.constant_(self.fc2.bias, init_lambda)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = F.relu(self.fc1(hidden_states))
        return self.fc2(hidden_states)


def construct_weight_ensembling_gate(
    hidden_size: int,
    num_experts: int,
    init_lambda: float,
    num_hidden_layers: int = 2,
):
    if num_hidden_layers == 0:
        gate = Depth_0_Gate(num_experts)
    elif num_hidden_layers == 1:
        gate = Depth_1_Gate(hidden_size, num_experts)
    elif num_hidden_layers == 2:
        gate = Depth_2_Gate(hidden_size, num_experts)
    else:
        raise ValueError(f"Unsupported number of hidden layers: {num_hidden_layers}")

    gate.num_hidden_layers = num_hidden_layers
    gate.init_weight(init_lambda)
    return gate


def positional_encoding(layer_idx, dim=8):
    """
    layer index encoding
    """
    pos = layer_idx
    i = np.arange(dim // 2)
    angle_rates = 1 / np.power(10000, (2 * i) / dim)
    encoding = np.concatenate([np.sin(pos * angle_rates), np.cos(pos * angle_rates)])
    return encoding


def _magnitude_prune(weight: Tensor, prune_ratio: float) -> Tensor:
    """
    Prune the weights by setting values below a certain quantile to zero.

    Args:
        weight (Tensor): The weight tensor to be pruned.
        prune_ratio (float): The ratio of weights to prune.

    Returns:
        Tensor: The pruned weight tensor.
    """
    weight_abs = weight.abs()
    mask = weight_abs > weight_abs.quantile(prune_ratio)
    weight = weight * mask
    return weight


def _module_magnitude_prune(
    model: Tensor, prune_ratio: float, layer_idx: int
) -> Tensor:
    """
    Prune a module.
    """
    for name, param in tqdm(
        model.named_parameters(),
        "Magnitude Pruning On {} Linear Layer".format(layer_idx),
        total=len(tuple(model.named_parameters())),
    ):
        param.data = _magnitude_prune(param, prune_ratio)
    return model


class SparseWeightEnsemblingMoE(nn.Module):
    # variable to store the merged state dict temporarily
    _merged_state_dict: StateDictType = None

    def __init__(
        self,
        hidden_size: int,
        base_model: nn.Module,
        expert_models: List[nn.Module],
        init_lambda: float = 0.2,
        batch_first: bool = False,
        router_hidden_layers: int = 2,
        batch_reduce: bool = False,
        num_layers: int = -1,
        layer_idx: int = -1,
        tv_prune_ratio: float = 0,
    ):
        """
        Initializes the SparseWeightEnsemblingMoE class.

        Args:

            hidden_size (int): The size of the hidden layer in the models.
            base_model (nn.Module): The base model that will be used as a reference for the expert models.
            expert_models (List[nn.Module]): A list of expert models that will be combined.
            init_lambda (float, optional): The initial lambda value for the weight ensembling gate. Defaults to 0.2.
            batch_first (bool, optional): If True, the input tensors are expected to have the batch size as the first dimension. Defaults to False.
            router_hidden_layers (int, optional): The number of hidden layers in the router. Defaults to 2.
            batch_reduce (bool): If True, the batch dimension of routing weights is reduced. Defaults to False.
            num_layers (int): Total number of layers
            layer_idx (int): Index of the layer
            tv_prune_ratio (int): What percentage of the parameters are removed
        """
        super().__init__()
        self.num_experts = len(expert_models)
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.batch_reduce = batch_reduce

        self.gate = construct_weight_ensembling_gate(
            hidden_size,
            self.num_experts,
            init_lambda=init_lambda,
            num_hidden_layers=router_hidden_layers,
        )

        # compute the task vectors
        for name, param in base_model.named_parameters():
            if not param.requires_grad:
                for m in expert_models:
                    del_attr(m, name.split("."))
            else:
                for m in expert_models:
                    get_attr(m, name.split(".")).data = (
                        get_attr(m, name.split(".")) - param
                    )

        # sparse task vectors
        expert_models = [
            _module_magnitude_prune(m, prune_ratio=tv_prune_ratio, layer_idx=layer_idx)
            for m in expert_models
        ]

        # fix base model and expert models
        self.base_model = base_model.requires_grad_(False)
        for m in expert_models:
            m.requires_grad_(False)
        self.task_vectors = nn.ModuleList(expert_models)

    @property
    def forward_model(self):
        return functools.partial(
            functional_call,
            self.base_model,
            self._merged_state_dict,
        )

    def merge_weights(self, expert_weights):
        state_dict = self.base_model.state_dict(keep_vars=True)
        for weight, task_vector in zip(expert_weights, self.task_vectors):
            for name, param in task_vector.named_parameters():
                state_dict[name] = state_dict[name] + weight * param
        self._merged_state_dict = state_dict
        return state_dict

    def forward(self, hidden_states: Tensor):
        if self.gate.num_hidden_layers == 0:
            gate_weights = self.gate()
        else:
            gate_weights = self.gate(hidden_states)
            if self.batch_first:
                # the input is in the shape of (batch_size, seq_len, hidden_size)
                gate_weights = gate_weights.mean(dim=1)
            else:
                # the input is in the shape of (seq_len, batch_size, hidden_size)
                gate_weights = gate_weights.mean(dim=0)

        if self.gate.num_hidden_layers == 0:
            self.merge_weights(gate_weights)
            output_hidden_states = self.forward_model(hidden_states)
        elif self.batch_reduce:
            gate_weights = gate_weights.mean(dim=0)
            self.merge_weights(gate_weights)
            output_hidden_states = self.forward_model(hidden_states)
        else:
            output_hidden_states = []
            for sample_idx, weights in enumerate(gate_weights):
                self.merge_weights(weights)
                if self.batch_first:
                    output_hidden_states.append(
                        self.forward_model(hidden_states[sample_idx : sample_idx + 1])
                    )
                else:
                    output_hidden_states.append(
                        self.forward_model(
                            hidden_states[:, sample_idx : sample_idx + 1]
                        )
                    )
            if self.batch_first:
                output_hidden_states = torch.cat(output_hidden_states, dim=0)
            else:
                output_hidden_states = torch.cat(output_hidden_states, dim=1)

        self._merged_state_dict = None
        return output_hidden_states


class SparseWeightEnsemblingMoE_ShardGate(nn.Module):
    # variable to store the merged state dict temporarily
    _merged_state_dict: StateDictType = None

    def __init__(
        self,
        hidden_size: int,
        base_model: nn.Module,
        expert_models: List[nn.Module],
        init_lambda: float = 0.2,
        batch_first: bool = False,
        router_hidden_layers: int = 2,
        batch_reduce: bool = False,
        num_layers: int = -1,
        layer_idx: int = -1,
        tv_prune_ratio: float = 0,
        sharedgate: nn.Module = None,
        position_encoding: bool = False,
        position_encoding_dim: int = 0,
    ):
        """
        Initializes the SparseWeightEnsemblingMoE class.

        Args:
            hidden_size (int): The size of the hidden layer in the models.
            base_model (nn.Module): The base model that will be used as a reference for the expert models.
            expert_models (List[nn.Module]): A list of expert models that will be combined.
            init_lambda (float, optional): The initial lambda value for the weight ensembling gate. Defaults to 0.2.
            batch_first (bool, optional): If True, the input tensors are expected to have the batch size as the first dimension. Defaults to False.
            router_hidden_layers (int, optional): The number of hidden layers in the router. Defaults to 2.
            batch_reduce (bool): If True, the batch dimension of routing weights is reduced. Defaults to False.
            num_layers (int): Total number of layers
            layer_idx (int): Index of the layer
            tv_prune_ratio (int): What percentage of the parameters are removed
            sharedgate (nn.Module): Shared routing mechanism
            position_encoding (bool): Is Positional Encoding enabled?
            position_encoding_dim (int): Positional Encoding dimension
        """
        super().__init__()
        self.num_experts = len(expert_models)
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.batch_reduce = batch_reduce
        self.position_encoding = position_encoding
        self.position_encoding_dim = position_encoding_dim
        # self.layer_idx = layer_idx

        self.gate = sharedgate
        if self.position_encoding:
            self.layer_positional_encoding = torch.from_numpy(
                positional_encoding(layer_idx, position_encoding_dim)
            ).float()

        # compute the task vectors
        for name, param in base_model.named_parameters():
            if not param.requires_grad:
                for m in expert_models:
                    del_attr(m, name.split("."))
            else:
                for m in expert_models:
                    get_attr(m, name.split(".")).data = (
                        get_attr(m, name.split(".")) - param
                    )

        # sparse task vectors
        expert_models = [
            _module_magnitude_prune(m, prune_ratio=tv_prune_ratio, layer_idx=layer_idx)
            for m in expert_models
        ]

        # fix base model and expert models
        self.base_model = base_model.requires_grad_(False)
        for m in expert_models:
            m.requires_grad_(False)
        self.task_vectors = nn.ModuleList(expert_models)

    @property
    def forward_model(self):
        return functools.partial(
            functional_call,
            self.base_model,
            self._merged_state_dict,
        )

    def merge_weights(self, expert_weights):
        state_dict = self.base_model.state_dict(keep_vars=True)
        for weight, task_vector in zip(expert_weights, self.task_vectors):
            for name, param in task_vector.named_parameters():
                state_dict[name] = state_dict[name] + weight * param
        self._merged_state_dict = state_dict
        return state_dict

    def forward(self, hidden_states: Tensor):
        gate_input = hidden_states

        if self.gate.num_hidden_layers == 0:
            gate_weights = self.gate()
        else:
            if self.position_encoding:
                layer_positional_encoding = (
                    self.layer_positional_encoding.unsqueeze(0)
                    .unsqueeze(0)
                    .expand(
                        hidden_states.size()[0],
                        hidden_states.size()[1],
                        self.position_encoding_dim,
                    )
                )
                layer_positional_encoding = layer_positional_encoding.to(
                    hidden_states.device
                )
                gate_input = torch.cat(
                    (layer_positional_encoding, hidden_states), dim=-1
                )

            gate_weights = self.gate(gate_input)
            if self.batch_first:
                # the input is in the shape of (batch_size, seq_len, hidden_size)
                gate_weights = gate_weights.mean(dim=1)
            else:
                # the input is in the shape of (seq_len, batch_size, hidden_size)
                gate_weights = gate_weights.mean(dim=0)

        # print('self.batch_reduce'+'-------------'+str(self.batch_reduce)+'-------------')
        if self.gate.num_hidden_layers == 0:
            self.merge_weights(gate_weights)
            output_hidden_states = self.forward_model(hidden_states)
        elif self.batch_reduce:
            gate_weights = gate_weights.mean(dim=0)
            self.merge_weights(gate_weights)
            output_hidden_states = self.forward_model(hidden_states)
        else:
            output_hidden_states = []
            for sample_idx, weights in enumerate(gate_weights):
                self.merge_weights(weights)
                if self.batch_first:
                    output_hidden_states.append(
                        self.forward_model(hidden_states[sample_idx : sample_idx + 1])
                    )
                else:
                    output_hidden_states.append(
                        self.forward_model(
                            hidden_states[:, sample_idx : sample_idx + 1]
                        )
                    )
            if self.batch_first:
                output_hidden_states = torch.cat(output_hidden_states, dim=0)
            else:
                output_hidden_states = torch.cat(output_hidden_states, dim=1)

        self._merged_state_dict = None
        return output_hidden_states
