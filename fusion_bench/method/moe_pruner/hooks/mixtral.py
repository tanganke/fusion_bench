from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .hook import BaseHookFn


class MoEPrunerHookFnForMixtralLinear(BaseHookFn):
    _routing_weights = None  # set by gate hook

    def __init__(
        self,
        linear: nn.Linear,
        name: str,
    ):
        super().__init__(linear)
        self.linear = linear
        self.scalar_row = torch.zeros(
            (linear.weight.size(1),), device=linear.weight.device
        )
        self.nsamples = 0
        self.name = name

    def compute(self):
        return torch.abs(self.linear.weight) * torch.sqrt(
            self.scalar_row.reshape(1, -1)
        )

    def __call__(self, linear: nn.Linear, inps: Tuple[Tensor], out: Tensor):
        assert len(inps) == 1
        inp = inps[0]
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        batch_size = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        # (NxL, C) -> (C, NxL)
        inp = inp.t()

        self.scalar_row *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size

        inp = inp.type(torch.float32)
        routing_weights = self._routing_weights.t()
        self.scalar_row += (
            torch.norm(inp * routing_weights, p=2, dim=1) ** 2 / self.nsamples
        )


class MoEPrunerHookFnForMixtralGate(BaseHookFn):

    def __init__(
        self,
        router: nn.Module,
        linear_layer_hooks: Dict[str, MoEPrunerHookFnForMixtralLinear],
        top_k: int,
        num_experts: int,
    ):
        self.nsamples = 0
        self.linear_layer_hooks = linear_layer_hooks
        self.top_k = top_k
        self.num_experts = num_experts
        super().__init__(router)

    def __call__(self, router, inps: Tuple[Tensor], out: Tensor):
        assert len(inps) == 1
        inp = inps[0]

        router_logits = out
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            for name, hook in self.linear_layer_hooks.items():
                if not name.startswith(f"{expert_idx}."):
                    continue
                hook._routing_weights = routing_weights[top_x, idx, None]

    def compute(self):
        pass
