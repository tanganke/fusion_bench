import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SmileMoEConfig:
    """
    Example PretrainedConfig for SmileMoE.

    Args:
        num_experts_per_tok: Number of experts per token.
        rank_of_router: Rank of the router.
        rank_of_expert: Rank of the expert.
        num_local_experts: Number of local experts.
    """

    num_experts_per_tok: int
    rank_of_router: int
    rank_of_expert: int
    num_local_experts: int


class SmileGate(nn.Module):
    __constants__ = ["in_features", "num_experts", "k"]
    in_features: int
    num_experts: int
    k: int
    weight: nn.Parameter

    def __init__(
        self,
        in_features: int,
        num_experts: int,
        k: int,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.input_features = in_features
        self.num_experts = num_experts
        self.k = k

        self.weight = nn.Parameter(
            torch.empty(num_experts * k, in_features, **factory_kwargs)
        )

    def forward(self, x: Tensor):
        batch_size = x.size(0)
        if self.num_experts == 1:
            return torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)

        routing_weights = F.linear(x, self.weight).view(
            batch_size, self.num_experts, self.k
        )
        routing_weights = routing_weights.norm(p=2, dim=2)
        return routing_weights


class SmileLinearExpert(nn.Module):
    __constants__ = ["in_features", "out_features", "k"]
    in_features: int
    out_features: int
    k: int

    def __init__(
        self,
        in_features,
        out_features,
        k: int,
        bias: bool,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k

        self.u = nn.Parameter(torch.empty(out_features, k, **factory_kwargs))
        self.svh = nn.Parameter(torch.empty(k, in_features, **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        x = F.linear(x, self.svh)
        x = F.linear(x, self.u, self.bias)
        return x


class SmileLinear(nn.Module):
    __constants__ = [
        "in_features",
        "out_features",
        "num_local_experts",
        "num_experts_per_tok",
        "rank_of_expert",
        "rank_of_router",
    ]

    in_features: int
    out_features: int
    num_local_experts: int
    num_experts_per_tok: int
    rank_of_expert: int
    rank_of_router: int

    @torch.no_grad()
    def __init__(
        self,
        config: SmileMoEConfig,
        in_features,
        out_features,
        bias: bool,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_local_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.rank_of_expert = config.rank_of_expert
        self.rank_of_router = config.rank_of_router
        self.in_features = in_features
        self.out_features = out_features

        # construct the gate network
        self.gate = SmileGate(
            in_features=in_features,
            num_experts=self.num_local_experts,
            k=self.rank_of_router,
            **factory_kwargs,
        )

        # the shared linear
        self.shared_linear = nn.Linear(
            in_features, out_features, bias=bias, **factory_kwargs
        )

        # construct experts
        if self.rank_of_expert > 0:
            self.experts = nn.ModuleList(
                [
                    SmileLinearExpert(
                        in_features=in_features,
                        out_features=out_features,
                        bias=bias,
                        k=self.rank_of_expert,
                        **factory_kwargs,
                    )
                    for _ in range(self.num_local_experts)
                ]
            )
        else:
            self.experts = nn.ModuleList(
                [
                    nn.Linear(in_features, out_features, bias=bias, **factory_kwargs)
                    for _ in range(self.num_local_experts)
                ]
            )

    def forward(self, hidden_states: Tensor):
        pretrained_out = self.shared_linear(hidden_states)

        input_shape = hidden_states.size()
        hidden_states = hidden_states.view(-1, self.in_features)

        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1)
        # sample the expert according to the routing weights
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
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
            selected_experts, num_classes=self.num_local_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_local_experts):
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
        return self.shared_linear.weight

    @property
    def bias(self):
        return self.shared_linear.bias

    def __repr__(self):
        return (
            f"SingularMoELinear("
            f"in_features={self.shared_linear.in_features}, "
            f"out_features={self.shared_linear.out_features}, "
            f"num_local_experts={self.num_local_experts}, "
            f"num_experts_per_tok={self.num_experts_per_tok}, "
            f"rank_of_router={self.rank_of_router}, "
            f"rank_of_expert={self.rank_of_expert}"
            f")"
        )
