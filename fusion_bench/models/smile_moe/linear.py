import logging
from typing import Dict, List, Tuple  # noqa: F401

import torch
import torch.nn.functional as F
from torch import Tensor, nn

log = logging.getLogger(__name__)


class ExpertNotTrainedError(Exception):
    pass


def _is_all_zeros(tensor: Tensor | List[Tensor]) -> bool:
    if isinstance(tensor, Tensor):
        return torch.allclose(tensor, torch.zeros_like(tensor))
    else:
        return all(_is_all_zeros(t) for t in tensor)


def _svd(w: Tensor, full_matrices=True) -> Tuple[Tensor, Tensor, Tensor]:
    u, s, vh = torch.linalg.svd(
        w, full_matrices=full_matrices, driver="gesvd" if w.is_cuda else None
    )
    v = vh.T
    return u, s, v


def svd(
    w: Tensor, full_matrices=True, accelerator=None
) -> Tuple[Tensor, Tensor, Tensor]:
    if accelerator is None:
        return _svd(w, full_matrices=full_matrices)
    original_device = w.device
    w = w.to(accelerator)
    u, s, v = _svd(w)
    return u.to(original_device), s.to(original_device), v.to(original_device)


class SmileGate(nn.Module):
    def __init__(
        self,
        input_features: int,
        w_diff_list: List[Tensor],
        k: int,
        svd_list=None,  # cached `svd_list`, pass it to avoid recomputing
        upscaling_accelerator=None,
    ):
        super().__init__()
        self.input_features = input_features
        self.num_experts = len(w_diff_list)
        weights = []
        for i, w_diff in enumerate(w_diff_list):
            if svd_list is None:
                u, s, v = svd(w_diff, accelerator=upscaling_accelerator)
            else:
                u, s, v = svd_list[i]
            u = u[:, :k]
            s = s[:k]
            v = v[:, :k]

            # weights.append((s * v).T)
            weights.append(v.T)
        self.k = s.size(0)  # k is the actual k after truncation

        weights = (
            torch.stack(weights, dim=0)
            .reshape(self.num_experts * self.k, -1)
            .contiguous()
        )
        self.weights = nn.Parameter(
            weights
        )  # weights should be a tensor of shape (num_experts * k, n)

    def forward(self, x: Tensor):
        batch_size = x.size(0)
        if self.num_experts == 1:
            return torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)

        routing_weights = F.linear(x, self.weights).view(
            batch_size, self.num_experts, self.k
        )
        routing_weights = routing_weights.norm(p=2, dim=2)
        return routing_weights


class SmileCompressedLinear(nn.Module):
    def __init__(self, model: nn.Linear, k: int, svd_cache=None):
        super().__init__()
        if svd_cache is None:
            u, s, v = svd(model.weight)
        else:
            u, s, v = svd_cache
        if k > 0:
            u = u[:, :k]
            s = s[:k]
            v = v[:, :k]

        self.u = nn.Parameter(u)
        self.svh = nn.Parameter((s * v).T)

        if model.bias is not None:
            self.bias = nn.Parameter(model.bias.data, requires_grad=True)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        x = F.linear(x, self.svh)
        x = F.linear(x, self.u, self.bias)
        return x


class SmileMoELinear(nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        pretrained_model: nn.Linear,
        finetuned_models: List[nn.Linear],
        gate_k: int,
        k: int,
        top_k: int = 1,
        full_matrices=True,
        upscaling_accelerator=None,
        routing_use_diff=True,
    ):
        super().__init__()
        self.num_experts = len(finetuned_models)
        self.top_k = top_k
        self.k = k
        self.gate_k = gate_k
        self.in_features = pretrained_model.in_features
        self.out_features = pretrained_model.out_features

        w_diff_list = [m.weight - pretrained_model.weight for m in finetuned_models]
        if _is_all_zeros(w_diff_list):
            # All fine-tuned models are identical to the pretrained model
            raise ExpertNotTrainedError()

        if routing_use_diff or k > 0:
            svd_cache_list = [
                svd(w, full_matrices=full_matrices, accelerator=upscaling_accelerator)
                for w in w_diff_list
            ]  # the svd cache list to avoid recomputing

        # construct the gate network
        if routing_use_diff:
            self.gate = SmileGate(
                input_features=self.in_features,
                w_diff_list=w_diff_list,
                k=gate_k,
                svd_list=svd_cache_list,
                upscaling_accelerator=upscaling_accelerator,
            )
        else:
            self.gate = SmileGate(
                input_features=self.in_features,
                w_diff_list=[m.weight for m in finetuned_models],
                k=gate_k,
                svd_list=None,
                upscaling_accelerator=upscaling_accelerator,
            )

        # construct experts
        for m, w_diff in zip(finetuned_models, w_diff_list):
            m.weight.data = w_diff
        if k > 0:
            experts = [
                SmileCompressedLinear(m, k, svd_cache=svd_cache)
                for m, svd_cache in zip(finetuned_models, svd_cache_list)
            ]
        else:
            # if k is not set (<0), we use the full fine-tuned model
            experts = finetuned_models
        self.experts = nn.ModuleList(experts)

        if pretrained_model.bias is not None:
            for m in experts:
                m.bias.data = m.bias.data - pretrained_model.bias
        # assign the pretrained model (the shared part)
        self.pretrained_model = pretrained_model

    def forward(self, hidden_states: Tensor):
        pretrained_out = self.pretrained_model(hidden_states)

        input_shape = hidden_states.size()
        hidden_states = hidden_states.view(-1, self.in_features)

        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1)
        # sample the expert according to the routing weights
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros(
            (hidden_states.size(0), self.out_features),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, self.in_features)
            if current_state.numel() == 0:
                continue
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(
            *input_shape[:-1], self.out_features
        )
        final_hidden_states = pretrained_out + final_hidden_states
        return final_hidden_states

    @property
    def weight(self):
        """
        Mimic linear layer. Bacause in some cases, user might indicate the device (or dtype of parameters) of the linear layer using `linear_layer.weight.device`
        """
        return self.pretrained_model.weight

    @property
    def bias(self):
        return self.pretrained_model.bias

    def __repr__(self):
        return (
            f"SingularMoELinear("
            f"in_features={self.pretrained_model.in_features}, "
            f"out_features={self.pretrained_model.out_features}, "
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"gate_k={self.gate_k}, "
            f"k={self.k}"
            f")"
        )
