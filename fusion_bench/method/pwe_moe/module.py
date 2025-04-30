R"""
this is adapted from
https://github.com/tanganke/weight-ensembling_MoE/blob/3cbd327cb28c499065f83387472a79829a2e5fee/src/module/dict_moe.py
but with some modifications
"""

import logging
from copy import deepcopy
from typing import List, Optional, cast

import torch
import torch.func
from torch import Tensor, nn
from torch.nn import functional as F

log = logging.getLogger(__name__)


def join_list(list_of_list: List[List]):
    ans = []
    for item in list_of_list:
        ans.extend(item)
    return ans


class PWEMoEGate(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        init_lambda: float,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        assert num_hidden_layers <= 2
        self.input_dim = hidden_size
        self.num_experts = num_experts
        self.num_hidden_layers = num_hidden_layers

        if num_hidden_layers == 2:
            self.fc1 = nn.Linear(hidden_size, hidden_size, bias=True)
            nn.init.normal_(self.fc1.weight, std=0.01)
            nn.init.zeros_(self.fc1.bias)
        elif num_hidden_layers == 1:
            self.fc1 = nn.Identity()

        if num_hidden_layers >= 1:
            self.fc2 = nn.Linear(hidden_size, num_experts, bias=True)
            nn.init.normal_(self.fc2.weight, std=0.01)
            nn.init.constant_(self.fc2.bias, init_lambda)

        if num_hidden_layers == 0:
            self.weight = nn.Parameter(
                torch.ones(num_experts) * init_lambda, requires_grad=True
            )

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.num_hidden_layers == 0:
            return self.weight

        if self.num_hidden_layers == 2:
            hidden_states = F.relu(self.fc1(hidden_states))
        gate_weights = self.fc2(hidden_states)
        return gate_weights


class PWEMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        base_model: nn.Module,
        expert_models: List[nn.Module],
        init_lambda: float = 0.2,
        fix_base_model_and_experts: bool = True,
        batch_first: bool = False,
        router_hidden_layers: int = 2,
    ):
        super().__init__()
        self.num_experts = len(expert_models)
        self.input_dim = hidden_size
        self.batch_first = batch_first

        self.gate = PWEMoEGate(
            hidden_size,
            self.num_experts,
            init_lambda=init_lambda,
            num_hidden_layers=router_hidden_layers,
        )

        self.base_model = deepcopy(base_model)
        experts = [deepcopy(e) for e in expert_models]
        base_sd = self.base_model.state_dict()
        experts_params = []
        experts_sd = [e.state_dict() for e in experts]
        for name in base_sd.keys():
            task_vectors = []
            for e_sd in experts_sd:
                with torch.no_grad():
                    _task_vector = e_sd[name] - base_sd[name]
                    task_vectors.append(_task_vector)
            task_vectors = torch.stack(task_vectors)
            experts_params.append(
                nn.Parameter(task_vectors, requires_grad=not fix_base_model_and_experts)
            )
        self.expert_parms = nn.ParameterList(experts_params)

        if fix_base_model_and_experts:
            for p in self.base_model.parameters():
                p.requires_grad_(False)
            for p in self.expert_parms.parameters():
                p.requires_grad_(False)

    def forward(self, hidden_states: Tensor):
        if not self.batch_first:
            hidden_states = hidden_states.permute(1, 0, 2)
        batch_size, seq_len, hidden_size = hidden_states.shape
        gate_weights: Tensor = self.gate(hidden_states)
        if self.gate.num_hidden_layers == 0:
            base_sd = self.base_model.state_dict(keep_vars=True)
            sd = {}
            for param_idx, (name, param) in enumerate(base_sd.items()):
                expert_params: nn.Parameter = self.expert_parms[param_idx]
                task_vector = expert_params * gate_weights.view(
                    [-1] + [1] * (expert_params.dim() - 1)
                )
                task_vector = task_vector.sum(dim=0)
                sd[name] = param + task_vector
            final_hidden_states = torch.func.functional_call(
                self.base_model, sd, hidden_states
            )
        else:
            gate_weights = gate_weights.mean(dim=1)
            final_hidden_states = []
            base_sd = self.base_model.state_dict(keep_vars=True)
            for sample_idx in range(batch_size):
                sd = {}
                for param_idx, (name, param) in enumerate(base_sd.items()):
                    expert_params: nn.Parameter = self.expert_parms[param_idx]
                    task_vector = expert_params * gate_weights[sample_idx].view(
                        [-1] + [1] * (expert_params.dim() - 1)
                    )
                    task_vector = task_vector.sum(dim=0)
                    sd[name] = param + task_vector
                _final_hidden_states = torch.func.functional_call(
                    self.base_model, sd, hidden_states[sample_idx : sample_idx + 1]
                )
                final_hidden_states.append(_final_hidden_states)
            final_hidden_states = torch.cat(final_hidden_states, dim=0)
        if not self.batch_first:
            final_hidden_states = final_hidden_states.permute(1, 0, 2)
        return final_hidden_states


class ParetoWeightEnsemblingModule(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        expert_models: List[nn.Module],
        init_lambda: float = 0.2,
        fix_base_model_and_experts: bool = True,
        router_hidden_layers: int = 1,
    ):
        super().__init__()
        self.num_experts = len(expert_models)

        # initialize the router, which is a simple MLP,
        # takes the preference vector as input and output the routing weights
        if router_hidden_layers == 1:
            self.gate = nn.Sequential(
                nn.Linear(self.num_experts, self.num_experts, bias=True),
            )
            nn.init.normal_(self.gate[0].weight, std=0.01)
            cast(nn.Parameter, self.gate[0].bias).data.fill_(init_lambda)
        elif router_hidden_layers == 2:
            self.gate = nn.Sequential(
                nn.Linear(self.num_experts, 2 * self.num_experts, bias=True),
                nn.ReLU(),
                nn.Linear(2 * self.num_experts, self.num_experts, bias=True),
            )
            nn.init.normal_(self.gate[0].weight, std=0.01)
            nn.init.zeros_(self.gate[0].bias)
            nn.init.normal_(self.gate[2].weight, std=0.01)
            cast(nn.Parameter, self.gate[2].bias).data.fill_(init_lambda)
        else:
            raise NotImplementedError()

        self.base_model = deepcopy(base_model)
        experts = [deepcopy(e) for e in expert_models]
        # state dict of the pre-trained model
        base_sd = self.base_model.state_dict()
        # state dict of the expert model
        expert_params = []
        experts_sd = [e.state_dict(keep_vars=True) for e in experts]
        # compute the task vector
        for name in base_sd.keys():
            task_vectors = []
            for e_sd in experts_sd:
                with torch.no_grad():
                    _task_vector = e_sd[name] - base_sd[name]
                    task_vectors.append(_task_vector)
            task_vectors = torch.stack(task_vectors)
            expert_params.append(
                nn.Parameter(task_vectors, requires_grad=not fix_base_model_and_experts)
            )

        self.expert_params = nn.ParameterList(expert_params)

        if fix_base_model_and_experts:
            self.base_model.requires_grad_(False)
            for p in self.expert_params.parameters():
                p.requires_grad_(False)

        self.preference_vector = None
        self._merged_state_dict = None

    def _set_preference_vector(self, perference_vector: Tensor):
        """
        Sets the preference vector for the model and resets the merged state dictionary cache.

        Args:
            preference_vector (Tensor): The preference vector to be set. It should be a 1D tensor
                                        with the same length as the number of experts.

        Raises:
            AssertionError: If the preference vector does not have the same length as the number of experts
                            or is not a 1D tensor.

        Returns:
            None
        """
        if not isinstance(perference_vector, Tensor):
            perference_vector = torch.as_tensor(perference_vector)
        self.preference_vector = perference_vector
        # reset the merged state dict cache
        self._merged_state_dict = None
        assert (
            self.preference_vector.shape[0] == self.num_experts
            and self.preference_vector.dim() == 1
        ), "preference vector should have the same length as the number of experts and be 1D tensor"

    def _merge_state_dict(self):
        assert self.preference_vector is not None, "preference vector is not set"
        routing_weights = self.gate(self.preference_vector)
        merged_state_dict = {}
        for param_idx, (name, params) in enumerate(
            self.base_model.state_dict(keep_vars=True).items()
        ):
            expert_params: nn.Parameter = self.expert_params[param_idx]
            task_vector = expert_params * routing_weights.view(
                [-1] + [1] * (expert_params.dim() - 1)
            )
            task_vector = task_vector.sum(dim=0)
            merged_state_dict[name] = params + task_vector
        return merged_state_dict

    def forward(self, *args, **kwargs):
        assert (
            self.preference_vector is not None
        ), "preference vector is not set, please call `set_preference_vector` before forward"
        if self._merged_state_dict is None:
            # cache the merged state dict
            self._merged_state_dict = self._merge_state_dict()
        return torch.func.functional_call(
            self.base_model, self._merged_state_dict, args=args, kwargs=kwargs
        )

    def get_merged_model(self):
        """
        merge the base model and the expert models according to the preference vector, return the merged model
        """
        merged_state_dict = self._merge_state_dict()
        model = deepcopy(self.base_model)
        model.load_state_dict(merged_state_dict)
        return model

    @staticmethod
    def set_preferenece_vector(model: nn.Module, preference_vector: Tensor):
        """
        Sets the preference vector for a given model. If the model is an instance of
        `ParetoWeightEnsemblingModule`, it directly sets the preference vector. Otherwise,
        it recursively sets the preference vector for all child modules.

        Args:
            model (nn.Module): The model for which the preference vector is to be set.
            preference_vector (Tensor): The preference vector to be set in the model.

        Returns:
            nn.Module: The model with the preference vector set.
        """
        if isinstance(model, ParetoWeightEnsemblingModule):
            model._set_preference_vector(preference_vector)
        for name, module in model.named_children():
            if isinstance(module, nn.Module):
                ParetoWeightEnsemblingModule.set_preferenece_vector(
                    module, preference_vector
                )
        return model

    @staticmethod
    def merge_and_unload(model: nn.Module):
        if isinstance(model, ParetoWeightEnsemblingModule):
            return model.get_merged_model()
        for name, module in model.named_children():
            if isinstance(module, nn.Module):
                setattr(
                    model, name, ParetoWeightEnsemblingModule.merge_and_unload(module)
                )
        return model

    def __repr__(self):
        return (
            f"ParetoWeightEnsemblingModule(base_model=<{type(self.base_model)}>, "
            f"num_expert_models={len(self.expert_params)}, "
            f"fix_base_model_and_experts={self.fix_base_model_and_experts}, "
            f"router_hidden_layers={self.router_hidden_layers})",
        )
