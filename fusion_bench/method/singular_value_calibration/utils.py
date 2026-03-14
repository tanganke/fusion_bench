"""
SVC Utilities: Interference Measurement and Singular Value Calibration

This module implements the core mathematical utilities from the paper:
"When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging"
(https://arxiv.org/abs/2602.05536)
"""

from typing import List, Optional

import torch


def subspace_consistency_spectral_calibration(
    base_weight: torch.Tensor,
    task_weights: List[torch.Tensor],
    merged_weight: torch.Tensor,
    alpha: float = 1.0,
    accelerator: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Perform subspace consistency spectral calibration (SVC) on the merged matrix.

    This implements Algorithm 1 from the paper "When Shared Knowledge Hurts:
    Spectral Over-Accumulation in Model Merging" (https://arxiv.org/abs/2602.05536).

    The implementation is fully vectorized to avoid Python-level loops over
    singular-vector indices, yielding significant speed improvements over a
    naive loop-based approach.

    Args:
        base_weight: Base model weight matrix W_{pre} of shape (m, n)
        task_weights: List of task-specific weight matrices W_i, each of shape (m, n)
        merged_weight: Initial merged weight matrix W_merge of shape (m, n)
        alpha: Scaling factor for calibration (default: 1.0). Acts as a floor
            for the projection coefficients so that the calibration never
            over-corrects a subspace.
        accelerator: Device to perform computations on. Defaults to CUDA/MPS
            when available, or the device of ``base_weight`` otherwise.

    Returns:
        Calibrated merged matrix W_calibrated of shape (m, n) on the same
        device as the input ``base_weight``.
    """
    if accelerator is None:
        if torch.cuda.is_available():
            accelerator = torch.device("cuda")
        elif torch.mps.is_available():
            accelerator = torch.device("mps")
        else:
            accelerator = base_weight.device

    original_device = base_weight.device

    base_weight = base_weight.to(accelerator)
    task_weights = [w.to(accelerator) for w in task_weights]
    merged_weight = merged_weight.to(accelerator)

    # ΔW_i = W_i - W_{pre}  and  ΔW_merge = W_merge - W_{pre}
    delta_task_matrices = [w - base_weight for w in task_weights]
    delta_merged = merged_weight - base_weight

    # Truncated SVD of the merged task matrix:
    #   U_merged : (m, k)   left singular vectors
    #   S_merged : (k,)     singular values
    #   Vh_merged: (k, n)   right singular vectors (conjugate-transposed)
    U_merged, S_merged, Vh_merged = torch.linalg.svd(
        delta_merged, full_matrices=False
    )

    # Project all matrices onto the shared left-singular-vector basis.
    # merged_responses[r] = U_merged[:, r]^T @ delta_merged  ->  shape (k, n)
    merged_responses = U_merged.T @ delta_merged  # (k, n)

    # task_responses[i, r] = U_merged[:, r]^T @ delta_task_matrices[i]
    # Stack into a single tensor of shape (K, k, n) for vectorised computation.
    K = len(delta_task_matrices)
    task_responses = torch.stack(
        [U_merged.T @ delta for delta in delta_task_matrices], dim=0
    )  # (K, k, n)

    # Projection coefficients (Eq. 8 from the paper):
    #   s_r^i = <a_r^merge, a_r^i> / ||a_r^i||^2
    #
    # Broadcast merged_responses (k, n) against task_responses (K, k, n).
    dot_products = (merged_responses * task_responses).sum(dim=-1)  # (K, k)
    norms_sq = (task_responses * task_responses).sum(dim=-1)  # (K, k)
    # Clamp denominator to avoid division by zero for zero-norm subspaces.
    proj_coeffs = dot_products / norms_sq.clamp(min=1e-10)  # (K, k)

    # Calibration factors (Eq. 9):
    #   gamma_r = K / sum_i( max(alpha, s_r^i) )
    gamma = K / proj_coeffs.clamp(min=alpha).sum(dim=0)  # (k,)

    # Reconstruct the calibrated delta matrix in the spectral domain and add
    # back the base weight.  We avoid torch.diag to keep memory overhead low.
    S_calibrated = gamma * S_merged  # (k,)
    delta_calibrated = (U_merged * S_calibrated.unsqueeze(0)) @ Vh_merged
    W_calibrated = delta_calibrated + base_weight

    return W_calibrated.to(original_device, non_blocking=True)
