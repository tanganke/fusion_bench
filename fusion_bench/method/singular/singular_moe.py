import os
import re
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import logging
from copy import deepcopy

from fusion_bench.method import ModelFusionAlgorithm
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import ModelPool, to_modelpool
from fusion_bench.utils.parameters import print_parameters

log = logging.getLogger(__name__)


class ExpertNotTrainedError(Exception):
    pass


def svd(w: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    u, s, vh = torch.linalg.svd(
        w, full_matrices=True, driver="gesvd" if w.is_cuda else None
    )
    v = vh.T
    return u, s, v


class Router(nn.Module):
    def __init__(
        self,
        input_features: int,
        w_diff_list: List[Tensor],
        k: int,
        svd_list=None,  # cached `svd_list`, pass it to avoid recomputing
    ):
        super().__init__()
        self.input_features = input_features
        self.num_experts = len(w_diff_list)
        weights = []
        for i, w_diff in enumerate(w_diff_list):
            if svd_list is None:
                u, s, v = svd(w_diff)
            else:
                u, s, v = svd_list[i]
            u = u[:, :k]
            s = s[:k]
            v = v[:, :k]

            # weights.append((s * v).T)
            weights.append((v).T)
        self.k = s.size(0)  # k is the actual k after truncation

        weights = torch.stack(weights, dim=0)
        self.weights = nn.Parameter(
            weights
        )  # weights should be a tensor of shape (num_experts, k, n)

    def forward(self, x: Tensor):
        batch_size = x.size(0)
        temp = torch.zeros(x.size(0), self.num_experts, self.k, device=x.device)
        for expert_idx in range(self.num_experts):
            expert_weights = self.weights[expert_idx]
            temp[:, expert_idx] = F.linear(x, expert_weights)
        r = temp  # (B, num_experts, k)
        r = r.norm(p=2, dim=2)  # (B, num_experts)
        return r


class BiasOnlyLinear(nn.Module):
    def __init__(self, model: nn.Linear):
        super().__init__()
        self.in_features = model.in_features
        self.out_features = model.out_features
        self.bias = nn.Parameter(model.bias.data, requires_grad=True)

    def forward(self, x):
        y = self.bias.expand(*x.size()[:-1], self.out_features)
        return y


class SingularCompressedLinear(nn.Module):
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
        return F.linear(x, self.u @ self.svh, self.bias)


def _is_all_zeros(tensor: Tensor | List[Tensor]) -> bool:
    if isinstance(tensor, Tensor):
        return torch.allclose(tensor, torch.zeros_like(tensor))
    else:
        return all(_is_all_zeros(t) for t in tensor)


class SingularMoELinear(nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        pretrained_model: nn.Linear,
        finetuned_models: List[nn.Linear],
        gate_k: int,
        k: int,
        top_k: int = 1,
        pretrained_bias_as_expert: bool = False,
    ):
        super().__init__()
        pretrained_bias_as_expert = (
            pretrained_bias_as_expert and pretrained_model.bias is not None
        )
        self.num_experts = len(finetuned_models) + int(pretrained_bias_as_expert)
        self.top_k = top_k
        self.in_features = pretrained_model.in_features
        self.out_features = pretrained_model.out_features

        for m in finetuned_models:
            m.weight.data = m.weight - pretrained_model.weight
        w_diff_list = [m.weight for m in finetuned_models]
        if _is_all_zeros(w_diff_list):
            # All fine-tuned models are identical to the pretrained model
            raise ExpertNotTrainedError()
        if pretrained_bias_as_expert:
            w_diff_list = [pretrained_model.weight] + w_diff_list
        svd_cache_list = [
            svd(w) for w in w_diff_list
        ]  # the svd cache list to avoid recomputing
        # construct the gate network
        self.gate = Router(
            input_features=self.in_features,
            w_diff_list=w_diff_list,
            k=gate_k,
            svd_list=svd_cache_list,
        )
        # construct experts
        experts = []
        if pretrained_bias_as_expert:
            svd_cache_list.pop(0)
            experts = [BiasOnlyLinear(pretrained_model)]
        if k > 0:
            experts = experts + [
                SingularCompressedLinear(m, k, svd_cache=svd_cache)
                for m, svd_cache in zip(finetuned_models, svd_cache_list)
            ]
        else:
            # if k is not set (<0), we use the full fine-tuned model
            experts = experts + finetuned_models
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


class SingularMoEUpscaling(ModelFusionAlgorithm, SimpleProfilerMixin):
    _linear_layer_cls = (nn.Linear,)

    @torch.no_grad()
    def run(self, modelpool: ModelPool):
        modelpool = to_modelpool(modelpool)

        if self.config.model_path is not None and os.path.exists(
            self.config.model_path
        ):
            log.info(f"Loading model from {self.config.model_path}")
            model = torch.load(self.config.model_path)
            print_parameters(model)
            return model

        with self.profile("load pretrained model"):
            pretrained_model = modelpool.load_model("_pretrained_")
        with self.profile("load fine-tuned model"):
            finetuned_models = [
                m for m in tqdm(modelpool.models(), total=len(modelpool.model_names))
            ]

        if self.config.device == "cuda" and torch.cuda.is_available():
            pretrained_model = pretrained_model.cuda()
            finetuned_models = [m.cuda() for m in finetuned_models]

        with self.profile("merge model"):
            model = self.merge(
                pretrained_model,
                finetuned_models,
                gate_k=self.config.gate_k,
                k=self.config.k,
                top_k=self.config.top_k,
                average_experts=self.config.average_experts,
            )

        self.print_profile_summary()
        if self.config.model_path is not None:
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            log.info(f"Saving model to {self.config.model_path}")
            torch.save(model, self.config.model_path)
        print_parameters(model)
        return model

    def merge(
        self,
        pretrained_model: nn.Module,
        finetuned_models: List[nn.Module],
        gate_k: int,
        k: int,
        top_k: int,
        average_experts: bool,
        in_place: bool = True,
    ):
        from fusion_bench.models.utils import get_attr, set_attr
        from fusion_bench.method.simple_average import simple_average

        if in_place:
            model = pretrained_model
        else:
            model = deepcopy(pretrained_model)

        for name, module in tqdm(
            tuple(model.named_modules()),
            "Upscaling Modules",
        ):
            if isinstance(module, self._linear_layer_cls):
                name_list = name.split(".")
                experts = [get_attr(m, name_list) for m in finetuned_models]
                try:
                    moe_linear = SingularMoELinear(
                        module, experts, gate_k=gate_k, k=k, top_k=top_k
                    )
                except ExpertNotTrainedError as e:
                    print(f"skip {name} because the experts are not trained.")
                    continue
                set_attr(model, name_list, moe_linear)
            elif average_experts and len(tuple(module.named_modules())) == 1:
                # if the module is a leaf module, we perform a parameter average
                name_list = name.split(".")
                experts = [get_attr(m, name_list) for m in finetuned_models]
                averaged_module = simple_average(experts)
                set_attr(model, name_list, averaged_module)
        return model
