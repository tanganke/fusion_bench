"""
This module contains the implementation of the Isotropic Merging in Common Subspace (ISO-C) algorithm and Isotropic Merging in Common and Task-Specific Subspaces (Iso-CTS) algorithm.
Modified from the original implementation: https://github.com/danielm1405/iso-merging

Reference:
- Daniel Marczak, et al. No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces. 2025.
    https://arxiv.org/abs/2502.04959
"""

from .iso import (
    ISO_C_Merge,
    ISO_CTS_Merge,
    IsotropicMergingInCommonAndTaskSubspace,
    IsotropicMergingInCommonSubspace,
)
