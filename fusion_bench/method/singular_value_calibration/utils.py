"""
SVC Utilities: Interference Measurement and Singular Value Calibration

This module implements the core mathematical utilities from the paper:
"When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging"
(https://arxiv.org/abs/2602.05536)
"""

from typing import Dict, List, Optional, Tuple

import torch


def project_onto_singular_vectors(
    task_matrix: torch.Tensor,
    U: torch.Tensor,
    num_subspaces: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    Project task matrix onto left singular vectors (column-space basis).

    This implements Eq (4) from the paper:
    a_r^i = u_r^T @ ΔW_i

    Each a_r^i captures the task's effect along the shared column-space
    direction u_r, producing a subspace response vector.

    Args:
        task_matrix: Task matrix ΔW_i of shape (m, n)
        U: Left singular vectors from SVD of merged task matrix, shape (m, m)
        num_subspaces: Number of top subspaces to compute (default: all)

    Returns:
        List of subspace responses a_r^i for each spectral subspace r
    """
    if num_subspaces is None:
        num_subspaces = U.shape[1]

    # Clip to available dimensions
    num_subspaces = min(num_subspaces, min(U.shape[1], task_matrix.shape[0]))

    subspace_responses = []
    for r in range(num_subspaces):
        u_r = U[:, r]  # Shape: (m,)
        # a_r^i = u_r^T @ ΔW_i -> Shape: (n,)
        a_r = u_r @ task_matrix  # Equivalent to u_r^T @ task_matrix
        subspace_responses.append(a_r)

    return subspace_responses


def compute_projection_coefficient(
    merged_response: torch.Tensor,
    task_response: torch.Tensor,
) -> torch.Tensor:
    """
    Compute projection coefficient that quantifies how the merged response
    scales along the task's response direction.

    This implements Eq (8) from the paper:
    s_r^i = <a_r^merge, a_r^i> / ||a_r^i||²

    Args:
        merged_response: Merged response a_r^merge in subspace r
        task_response: Task-specific response a_r^i in subspace r

    Returns:
        Projection coefficient s_r^i
        - s_r^i > 1: merged response amplifies task direction (over-counting)
        - s_r^i < 1: merged response attenuates task direction
        - s_r^i = 1: ideal case, preserved
    """
    norm_squared = torch.dot(task_response, task_response)
    projection = torch.dot(merged_response, task_response)
    s_r = projection / norm_squared
    return s_r


def compute_interference(
    merged_response: torch.Tensor,
    task_response: torch.Tensor,
) -> torch.Tensor:
    """
    Compute interference - the mismatch between merged and task-specific responses.

    This implements Eq (7) from the paper:
    I_r^i = ||Proj(a_r^merge) - a_r^i||²

    Where Proj(a_r^merge) = s_r^i * a_r^i is the projection of merged response
    onto task i's response direction.

    Args:
        merged_response: Merged response a_r^merge in subspace r
        task_response: Task-specific response a_r^i in subspace r

    Returns:
        Interference I_r^i (scalar, >= 0)
    """
    s_r = compute_projection_coefficient(merged_response, task_response)

    # Proj(a_r^merge) = s_r^i * a_r^i
    projected_response = s_r * task_response

    # I_r^i = ||Proj(a_r^merge) - a_r^i||²
    interference = torch.dot(
        projected_response - task_response, projected_response - task_response
    )

    return interference


def subspace_consistency_spectral_calibration(
    base_weight: torch.Tensor,
    task_weights: List[torch.Tensor],
    merged_weight: torch.Tensor,
    alpha: float = 1.0,
    accelerator: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Perform subspace consistency spectral calibration (SVC) on the merged matrix.

    This implements Algorithm 1 from the paper.

    Args:
        base_weight: Base model weight matrix W_{pre} of shape (m, n)
        task_weights: List of task-specific weight matrices W_i, each of shape (m, n)
        merged_weight: Initial merged weight matrix W_merge of shape (m, n)
        alpha: Scaling factor for calibration (default: 1.0)
        accelerator: Device to perform computations on (default: None, uses CUDA if available)

    Returns:
        Calibrated merged matrix W_calibrated of shape (m, n)
    """
    if accelerator is None:
        accelerator = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_device = base_weight.device

    base_weight = base_weight.to(accelerator)
    task_weights = [task_weight.to(accelerator) for task_weight in task_weights]
    merged_weight = merged_weight.to(accelerator)

    # Compute ΔW_i = W_i - W_{pre} for each task
    delta_task_matrices = [task_weight - base_weight for task_weight in task_weights]
    # Compute ΔW_merge = W_merge - W_{pre}
    delta_merged_matrix = merged_weight - base_weight

    # Project each task's ΔW_i onto the singular vectors U to get subspace responses
    task_responses = []
    for i in range(len(delta_task_matrices)):
        U, _, _ = torch.svd(delta_task_matrices[i])
        task_responses.append(
            project_onto_singular_vectors(task_matrix=delta_task_matrices[i], U=U)
        )
        del U  # Free memory

    # Project ΔW_merge onto the singular vectors U to get merged responses
    U_merged, S_merged, V_merged = torch.svd(delta_merged_matrix)
    merged_responses = project_onto_singular_vectors(
        task_matrix=delta_merged_matrix, U=U_merged
    )

    # Compute projection coefficients and interference for each task and subspace
    projection_coefficients = []
    for i in range(len(task_responses)):
        task_proj_coeffs = []
        for r in range(len(merged_responses)):
            s_r = compute_projection_coefficient(
                merged_response=merged_responses[r], task_response=task_responses[i][r]
            )
            task_proj_coeffs.append(s_r)
        projection_coefficients.append(task_proj_coeffs)

    # If multiple tasks contribute constructively in the same subspace, then sr  i > 1, indicating over-counting.
    # To produce a single correction per subspace, we aggregate these coefficients across tasks into a calibration factor
    calibration_factors = []
    K = len(task_responses)  # Number of tasks
    for r in range(len(merged_responses)):
        gamma_r = K / sum(
            max(alpha, s_r) for s_r in [projection_coefficients[i][r] for i in range(K)]
        )
        calibration_factors.append(gamma_r)

    # Apply calibration factors to the merged matrix in the spectral domain
    S_calibrated = torch.tensor(calibration_factors) * S_merged
    delta_merged_matrix_calibrated = U_merged @ torch.diag(S_calibrated) @ V_merged.t()
    W_calibrated = delta_merged_matrix_calibrated + base_weight

    return W_calibrated.to(original_device, non_blocking=True)
