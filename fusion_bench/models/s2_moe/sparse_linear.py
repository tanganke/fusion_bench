import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from fusion_bench.method.pruning.prune_utils import unstructured_magnitude_prune_


class SparseLinear(nn.Linear):
    """
    A sparse linear layer that can be used to compress the model.
    It will automatically convert the weight matrix to a sparse matrix after pruning.

    Example:

    >>> import torch
    >>> from fusion_bench.models.s2_moe.sparse_linear import SparseLinear

    >>> linear = SparseLinear(in_features, out_features, sparsity=0.5, bias=True)
    >>> linear.apply_pruning_()

    >>> input = torch.randn(10, in_features)
    >>> output = linear(input)
    >>> print(output)
    """

    __constants__ = ["in_features", "out_features", "sparsity_ratio"]

    in_features: int
    out_features: int
    sparsity_ratio: float

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        sparsity_ratio=0.5,
        device=None,
        dtype=None,
    ):
        """
        Args:
            in_features (int): Size of each input sample
            out_features (int): Size of each output sample
            sparsity (float): Target sparsity level (fraction of weights to be zero), between 0 and 1
            bias (bool): If set to False, the layer will not learn an additive bias
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity_ratio = sparsity_ratio
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def apply_pruning_(self):
        """
        Apply unstructured pruning to the weight matrix and convert the weight matrix to a sparse matrix
        """
        unstructured_magnitude_prune_(
            self.weight,
            torch.abs,
            sparsity_ratio=self.sparsity_ratio,
            return_pruned_weight=False,
        )
        self.weight = nn.Parameter(
            self.weight.to_sparse(), requires_grad=self.weight.requires_grad
        )

    def forward(self, input: torch.Tensor):
        if self.weight.is_sparse:
            y = torch.sparse.mm(input, self.weight.t())
            y = y + self.bias
            return y
        else:
            return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        """Additional information to print"""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"sparsity_ratio={self.sparsity_ratio}"
        )


if __name__ == "__main__":
    # test the sparse linear layer
    in_features = 10
    out_features = 20
    sparsity_ratio = 0.5
    bias = True
    sparse_linear = SparseLinear(in_features, out_features, sparsity_ratio, bias)
    sparse_linear.apply_pruning_()

    print(sparse_linear.weight)
    print(sparse_linear.bias)
    print(sparse_linear.weight.is_sparse)

    input = torch.randn(10, in_features)
    output = sparse_linear(input)
    print(output)
