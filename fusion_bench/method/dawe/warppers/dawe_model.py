import functools
from typing import List, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call

from fusion_bench.models.utils import del_attr, get_attr
from fusion_bench.utils.devices import get_device
from fusion_bench.utils.state_dict_arithmetic import (
    StateDictType,
    state_dict_weighted_sum,
)

from ....models.wrappers.task_wise_fusion import TaskWiseMergedModel


class Depth_0_Gate(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_dim), requires_grad=True)

    def init_weight(self, init_lambda: float):
        nn.init.constant_(self.weight, init_lambda)

    def forward(self, *args, **kwargs) -> Tensor:
        return self.weight


class Depth_1_Gate(nn.Module):
    def __init__(self, hidden_size: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, output_dim, bias=True)

    def init_weight(self, init_lambda: float):
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, init_lambda)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.fc(hidden_states)


class Depth_2_Gate(nn.Module):
    def __init__(self, hidden_size: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, output_dim, bias=True)

    def init_weight(self, init_lambda: float):
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.constant_(self.fc2.bias, init_lambda)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = F.relu(self.fc1(hidden_states))
        return self.fc2(hidden_states)


def construct_dawe_gate(
    hidden_size: int,
    coding_size: int,
    init_lambda: float,
    num_hidden_layers: int = 2,
):
    if num_hidden_layers == 0:
        gate = Depth_0_Gate(coding_size)
    elif num_hidden_layers == 1:
        gate = Depth_1_Gate(hidden_size, coding_size)
    elif num_hidden_layers == 2:
        gate = Depth_2_Gate(hidden_size, coding_size)
    else:
        raise ValueError(f"Unsupported number of hidden layers: {num_hidden_layers}")

    gate.num_hidden_layers = num_hidden_layers
    gate.init_weight(init_lambda)
    return gate


class DataAdaptiveWeightEnsemblingModel(nn.Module):

    def __init__(
        self,
        *,
        merge_mode: Literal["task_wise", "layer_wise"],
        hidden_size: int,
        dict_processor,
        model_processor,
        collate_fn=torch.stack,
        dict_feature_extractor: nn.Module,
        base_model: nn.Module,
        expert_models: List[nn.Module],
        init_lambda: float = 0.2,
        gate_hidden_layers: int = 2,
        batch_reduce: bool = False,
    ):
        super().__init__()
        self.merge_mode = merge_mode
        self.batch_reduce = batch_reduce
        self.num_experts = len(expert_models)

        self.collate_fn = collate_fn
        self.dict_processor = dict_processor
        self.model_processor = model_processor
        self.dict_feature_exactor = dict_feature_extractor
        if isinstance(self.dict_feature_exactor, nn.Module):
            self.dict_feature_exactor.requires_grad_(False)  # fix the feature extractor
        self.base_model = base_model

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
        # fix base model and expert models
        self.base_model = base_model.requires_grad_(False)
        for m in expert_models:
            m.requires_grad_(False)
        self.task_vectors = nn.ModuleList(expert_models)
        self.num_layers = len(self.task_vectors[0].state_dict())

        if self.merge_mode == "task_wise":
            self.coding_size = self.num_experts
        elif self.merge_mode == "layer_wise":
            self.coding_size = self.num_experts * self.num_layers
        else:
            raise ValueError(
                "Invalid option of `merge_model`, must be 'task_wise' or 'layer_wise'"
            )

        self.gate = construct_dawe_gate(
            hidden_size,
            coding_size=self.coding_size,
            init_lambda=init_lambda,
            num_hidden_layers=gate_hidden_layers,
        )

    def compute_task_vectors(self, coding_weights: Tensor):
        if self.merge_mode == "task_wise":
            state_dict = state_dict_weighted_sum(
                [
                    task_vector.state_dict(keep_vars=True)
                    for task_vector in self.task_vectors
                ],
                coding_weights,
            )
        elif self.merge_mode == "layer_wise":
            coding_weights = coding_weights.view(self.num_experts, -1)
            state_dict = {}
            for weight, task_vector in zip(coding_weights, self.task_vectors):
                for name, param in task_vector.state_dict(keep_vars=True).items():
                    state_dict[name] = state_dict.get(name, 0) + weight * param
        else:
            raise ValueError(
                "Invalid option of `merge_model`, must be 'task_wise' or 'layer_wise'"
            )
        return state_dict

    def merge_weights(self, task_vector: StateDictType):
        state_dict = self.base_model.state_dict(keep_vars=True)
        for name, param in task_vector.items():
            state_dict[name] = state_dict[name] + param
        return state_dict

    def model_forward(self, dict_codings, *args, **kwargs):
        if self.batch_reduce:
            dict_codings = dict_codings.mean(dim=0)
            task_vector = self.compute_task_vectors(dict_codings)
            state_dict = self.merge_weights(task_vector)
            return functional_call(
                self.base_model, state_dict, args=args, kwargs=kwargs, strict=True
            )
        else:
            model_outputs = []
            for dict_coding in dict_codings:
                task_vector = self.compute_task_vectors(dict_coding)
                state_dict = self.merge_weights(task_vector)
                model_outputs.append(
                    functional_call(
                        self.base_model,
                        state_dict,
                        args=args,
                        kwargs=kwargs,
                        strict=True,
                    )
                )
            model_outputs = self.collate_fn(model_outputs)
            return model_outputs

    def forward(self, *args, **kwargs):
        # compute dict codings
        if self.dict_processor is not None:
            inputs = self.dict_processor(*args, **kwargs)
            dict_features = self.dict_feature_exactor(inputs)
        else:
            dict_features = self.dict_feature_exactor(*args, **kwargs)
        dict_codings: Tensor = self.gate(dict_features)

        if self.model_processor is not None:
            inputs = self.model_processor(*args, **kwargs)
            model_outputs = self.model_forward(dict_codings, inputs)
        else:
            model_outputs = self.model_forward(dict_codings, *args, **kwargs)
        return model_outputs
