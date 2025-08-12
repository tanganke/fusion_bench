import functools
import logging
from typing import Dict, List, Tuple  # noqa: F401

import torch
import torch.func
from torch import Tensor, nn
from torch.func import functional_call
from torch.nn import functional as F

from fusion_bench.models.smile_moe.utils import _is_all_zeros, svd
from fusion_bench.models.utils import del_attr, get_attr, set_attr
from fusion_bench.utils.type import StateDictType

log = logging.getLogger(__name__)


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
        self.fc1 = nn.Linear(hidden_size, num_experts * 2, bias=True)
        self.fc2 = nn.Linear(num_experts * 2, num_experts, bias=True)

    def init_weight(self, init_lambda: float):
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.constant_(self.fc2.bias, init_lambda)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = F.relu(self.fc1(hidden_states))
        return self.fc2(hidden_states)


def construct_rankone_moe_gate(
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


class ExpertNotTrainedError(Exception):
    pass


def fun_joint_svd(
    w_list: List[Tensor], accelerator=None
) -> Tuple[Tensor, Tensor, Tensor]:

    w = torch.cat(w_list, dim=1)  # stacked_matrix
    original_device = w.device
    if accelerator is not None:
        w = w.to(accelerator)
    u_c, s_c, vh_c = torch.linalg.svd(
        w, full_matrices=False, driver="gesvd" if w.is_cuda else None
    )

    svd_list = []
    offset = 0
    for matrix in w_list:
        n_cols = matrix.size(1)
        u = u_c
        s = s_c
        vh_ = vh_c[:, offset : offset + n_cols]
        v = vh_.T
        svd_list.append(
            [u.to(original_device), s.to(original_device), v.to(original_device)]
        )

        offset += n_cols
    return svd_list


class RankOneMoE(nn.Module):
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
        svd_accelerator=False,
        rank_k: int = -1,
        select_k: int = -1,
    ):
        """
        Initializes the RankOneMoE class.
        https://github.com/EnnengYang/RankOne-MoE

        Args:
            hidden_size (int): The size of the hidden layer in the models.
            base_model (nn.Module): The base model that will be used as a reference for the expert models.
            expert_models (List[nn.Module]): A list of expert models that will be combined.
            init_lambda (float, optional): The initial lambda value for the weight ensembling gate. Defaults to 0.2.
            batch_first (bool, optional): If True, the input tensors are expected to have the batch size as the first dimension. Defaults to False.
            router_hidden_layers (int, optional): The number of hidden layers in the router. Defaults to 2.
            batch_reduce (bool): If True, the batch dimension of routing weights is reduced. Defaults to False.
        """
        super().__init__()
        self.num_experts = len(expert_models)
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.batch_reduce = batch_reduce
        self.svd_accelerator = svd_accelerator
        self.rank_k = rank_k
        self.select_k = select_k
        self.init_lambda = init_lambda

        self.gate = construct_rankone_moe_gate(
            hidden_size=hidden_size,
            num_experts=int(self.num_experts * self.rank_k),
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

        # fix base model and expert models
        self.base_model = base_model.requires_grad_(False)
        for m in expert_models:
            m.requires_grad_(False)

        # task vecotr  (only bias term)
        self.task_vectors_fc1_bias = nn.Parameter(
            torch.stack([e.fc1.bias for e in expert_models], dim=0), requires_grad=False
        )
        self.task_vectors_fc2_bias = nn.Parameter(
            torch.stack([e.fc2.bias for e in expert_models], dim=0), requires_grad=False
        )

        # SVD representation of task vector (only weight term)
        self.task_vectors_fc1_u = nn.ParameterList()
        self.task_vectors_fc1_svh = nn.ParameterList()
        self.task_vectors_fc2_u = nn.ParameterList()
        self.task_vectors_fc2_svh = nn.ParameterList()

        for m in expert_models:
            for name, param in m.named_parameters():
                if ".weight" in name:

                    if _is_all_zeros(param):
                        # All fine-tuned models are identical to the pretrained model
                        raise ExpertNotTrainedError()

                    u, s, v = svd(param, accelerator=self.svd_accelerator)
                    u = u[:, : self.rank_k]
                    s = s[: self.rank_k]
                    v = v[:, : self.rank_k]

                    if "fc1.weight" == name:
                        self.task_vectors_fc1_u.append(
                            nn.Parameter(u.T, requires_grad=False)
                        )
                        self.task_vectors_fc1_svh.append(
                            nn.Parameter((s * v).T, requires_grad=False)
                        )
                    elif "fc2.weight" == name:
                        self.task_vectors_fc2_u.append(
                            nn.Parameter(u.T, requires_grad=False)
                        )
                        self.task_vectors_fc2_svh.append(
                            nn.Parameter((s * v).T, requires_grad=False)
                        )

        # remove the original module from fine-tuned models to save memory
        for name, param in base_model.named_parameters():
            name_list = name.split(".")
            for m in expert_models:
                set_attr(m, name_list, None)

    @property
    def forward_model(self):
        return functools.partial(
            functional_call,
            self.base_model,
            self._merged_state_dict,
        )

    def top_k_soft(self, s, k):
        threshold, _ = torch.topk(s, k, largest=True, sorted=False)
        min_threshold = threshold.min()
        # sigmoid -> mask
        mask = torch.sigmoid(100 * (s - min_threshold))
        result = s * mask
        return result

    def merge_weights(self, expert_weights):
        state_dict = self.base_model.state_dict(keep_vars=True)

        # Select top-k experts from the expert pool for fusion
        if self.select_k > 0:
            expert_weights = self.top_k_soft(expert_weights, self.select_k)

        for name in state_dict:
            if name == "fc1.bias":
                for param in self.task_vectors_fc1_bias:
                    state_dict[name] = state_dict[name] + self.init_lambda * param
            elif name == "fc2.bias":
                for param in self.task_vectors_fc2_bias:
                    state_dict[name] = state_dict[name] + self.init_lambda * param

            elif name == "fc1.weight":
                w_list = torch.split(
                    expert_weights,
                    int(expert_weights.size(-1) / self.num_experts),
                    dim=-1,
                )
                for weight, u, svh in zip(
                    w_list, self.task_vectors_fc1_u, self.task_vectors_fc1_svh
                ):
                    weight_diag = torch.diag(weight)
                    weight_u = torch.mm(weight_diag, u)
                    result = torch.matmul(weight_u.T, svh)
                    state_dict[name] = state_dict[name] + result

            elif name == "fc2.weight":
                w_list = torch.split(
                    expert_weights,
                    int(expert_weights.size(-1) / self.num_experts),
                    dim=-1,
                )
                for weight, u, svh in zip(
                    w_list, self.task_vectors_fc2_u, self.task_vectors_fc2_svh
                ):
                    weight_diag = torch.diag(weight)
                    weight_u = torch.mm(weight_diag, u)
                    result = torch.matmul(weight_u.T, svh)
                    state_dict[name] = state_dict[name] + result

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
