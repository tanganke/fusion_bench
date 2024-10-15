import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter, init


class LoSparseLinear(nn.Module):
    skip_lowrank: bool = False
    skip_sparse: bool = False

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.lo_A = Parameter(torch.empty((rank, in_features), **factory_kwargs))
        self.lo_B = Parameter(torch.empty((out_features, rank), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.lo_B, a=math.sqrt(5))
        init.kaiming_uniform_(self.lo_A, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if self.skip_lowrank:
            low_rank_out = 0
        else:
            low_rank_out = F.linear(F.linear(input, self.lo_A), self.lo_B)
        if self.skip_sparse:
            sparse_out = self.bias if self.bias is not None else 0
        else:
            sparse_out = F.linear(input, self.weight, self.bias)
        return sparse_out + low_rank_out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, bias={self.bias is not None}"
