R"""
Copy from https://github.com/Lucky-Lance/Expert_Sparsity/tree/main/model

Original repo: https://github.com/Lucky-Lance/Expert_Sparsity

Reference:
    Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models.
    ACL 2024.
    http://arxiv.org/abs/2402.14800
"""

from .wrapper import (
    PrunableMixtralSparseMoeBlockWrapper,
    DynamicSkippingMixtralSparseMoeBlockWrapper,
)
