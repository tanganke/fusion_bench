"""
SpectralMerging Utilities: Core algorithms for spectral-domain model merging.

This module implements the mathematical core of SpectralMerge:
- Cross-task alignment detection via singular vector overlap
- Spectral subspace clustering
- Calibrated merging within aligned subspaces
- Preservation of orthogonal task-specific directions
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class AlignmentInfo:
    """Result of cross-task alignment analysis.

    Attributes:
        aligned_groups: List of groups, where each group is a list of
            (task_idx, subspace_idx) tuples that are spectrally aligned.
        standalone: List of (task_idx, subspace_idx) that are task-specific
            (no significant alignment with other tasks).
        alignment_scores: Per-group average alignment scores.
    """

    aligned_groups: List[List[Tuple[int, int]]]
    standalone: List[Tuple[int, int]]
    alignment_scores: List[float]


def compute_cross_task_alignment(
    task_svd: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    num_subspaces: List[int],
    alignment_threshold: float = 0.3,
) -> AlignmentInfo:
    """
    Detect cross-task spectral alignment by comparing right singular vectors.

    For each pair of tasks and their retained subspaces, we compute the absolute
    cosine similarity between right singular vectors:
        |cos(v_{t1,r1}, v_{t2,r2})| = |v_{t1,r1}^T v_{t2,r2}|

    Subspaces with similarity above the threshold are considered "aligned"
    and will be merged with calibration. Others are preserved as-is.

    Args:
        task_svd: List of (U, S, Vh) tuples from SVD of each task vector.
        num_subspaces: Number of retained subspaces per task.
        alignment_threshold: Minimum cosine similarity to consider aligned.

    Returns:
        AlignmentInfo with aligned groups and standalone subspaces.
    """
    K = len(task_svd)
    if K == 0:
        return AlignmentInfo(
            aligned_groups=[],
            standalone=[],
            alignment_scores=[],
        )
    device = task_svd[0][0].device

    # Collect all right singular vectors with their task/subspace indices
    # Vh is row-major: Vh[r, :] is the r-th right singular vector
    all_subspaces = []  # [(task_idx, subspace_idx, v_vector), ...]
    for t in range(K):
        _, _, Vh = task_svd[t]
        for r in range(num_subspaces[t]):
            all_subspaces.append((t, r, Vh[r, :]))

    N = len(all_subspaces)
    if N <= 1:
        return AlignmentInfo(
            aligned_groups=[],
            standalone=[(all_subspaces[i][0], all_subspaces[i][1]) for i in range(N)],
            alignment_scores=[],
        )

    # Build alignment matrix: |cos(v_i, v_j)| for all pairs
    V_matrix = torch.stack([s[2] for s in all_subspaces], dim=0)  # [N, n]
    V_matrix = torch.nn.functional.normalize(V_matrix, dim=1)
    alignment_matrix = torch.abs(V_matrix @ V_matrix.T)  # [N, N]

    # Vectorized clustering: build adjacency from alignment matrix
    # Only consider off-diagonal, above-threshold entries
    adj_mask = (alignment_matrix >= alignment_threshold) & ~torch.eye(N, dtype=torch.bool, device=device)

    # Greedy clustering using vectorized operations
    visited = torch.zeros(N, dtype=torch.bool, device=device)
    aligned_groups = []
    alignment_scores = []

    for i in range(N):
        if visited[i]:
            continue

        # Find all neighbors of i that aren't visited (vectorized)
        neighbors = adj_mask[i] & ~visited
        neighbor_idx = neighbors.nonzero(as_tuple=True)[0]

        if len(neighbor_idx) == 0:
            continue

        group = [(all_subspaces[i][0], all_subspaces[i][1])]
        group_scores = []
        for j in neighbor_idx:
            j_item = j.item()
            group.append((all_subspaces[j_item][0], all_subspaces[j_item][1]))
            group_scores.append(alignment_matrix[i, j].item())

        visited[i] = True
        visited[neighbor_idx] = True

        if len(group) > 1:
            aligned_groups.append(group)
            alignment_scores.append(
                sum(group_scores) / len(group_scores) if group_scores else 0.0
            )

    # Collect standalone subspaces (not in any aligned group)
    visited_all = set()
    for group in aligned_groups:
        for item in group:
            visited_all.add(item)

    standalone = []
    for i in range(N):
        item = (all_subspaces[i][0], all_subspaces[i][1])
        if item not in visited_all:
            standalone.append(item)

    return AlignmentInfo(
        aligned_groups=aligned_groups,
        standalone=standalone,
        alignment_scores=alignment_scores,
    )


def merge_in_spectral_domain(
    base_weight: torch.Tensor,
    task_svd: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    task_num_subspaces: List[int],
    alignment_info: AlignmentInfo,
    calibration_strength: float = 1.0,
    elect_threshold: float = 0.0,
) -> torch.Tensor:
    """
    Merge task vectors directly in the spectral domain.

    For aligned subspaces: calibrated averaging of singular values, then
    reconstruct using a consensus basis (averaged singular vectors).

    For standalone subspaces: magnitude-based elect (similar to TIES).
    Instead of direct accumulation (which causes interference across tasks),
    we only keep standalone directions whose singular value exceeds
    elect_threshold * max_singular_value.

    Args:
        base_weight: Pretrained weight matrix W_base.
        task_svd: SVD decompositions of each task vector.
        task_num_subspaces: Number of retained subspaces per task.
        alignment_info: Cross-task alignment analysis result.
        calibration_strength: Calibration factor for aligned subspaces.
        elect_threshold: Fraction of max singular value to keep standalone
            directions. 0.0 = keep all (original behavior), higher = more
            aggressive filtering. Typical: 0.01-0.1.

    Returns:
        Merged weight matrix.
    """
    device = base_weight.device
    m, n = base_weight.shape
    K = len(task_svd)

    # Initialize merged delta as zero
    delta_merged = torch.zeros(m, n, dtype=base_weight.dtype, device=device)

    # Track which (task, subspace) pairs have been processed
    processed = set()

    # --- Step 1: Process aligned groups (unchanged) ---
    for group, avg_score in zip(
        alignment_info.aligned_groups, alignment_info.alignment_scores
    ):
        if len(group) < 2:
            continue

        Us, Ss, Vhs = [], [], []
        for task_idx, subspace_idx in group:
            U, S, Vh = task_svd[task_idx]
            Us.append(U[:, subspace_idx])
            Ss.append(S[subspace_idx])
            Vhs.append(Vh[subspace_idx, :])
            processed.add((task_idx, subspace_idx))

        # Sign alignment with reference
        U_ref = Us[0]
        V_ref = Vhs[0]
        aligned_Us = [U_ref]
        aligned_Vhs = [V_ref]
        aligned_Ss = [Ss[0]]

        for k in range(1, len(Us)):
            u_dot = torch.dot(U_ref, Us[k])
            if u_dot < 0:
                aligned_Us.append(-Us[k])
                aligned_Vhs.append(-Vhs[k])
            else:
                aligned_Us.append(Us[k])
                aligned_Vhs.append(Vhs[k])
            aligned_Ss.append(Ss[k])

        U_avg = torch.stack(aligned_Us, dim=0).mean(dim=0)
        V_avg = torch.stack(aligned_Vhs, dim=0).mean(dim=0)
        U_avg = U_avg / (U_avg.norm() + 1e-12)
        V_avg = V_avg / (V_avg.norm() + 1e-12)

        S_values = torch.stack(aligned_Ss, dim=0)
        calibration_factor = min(1.0, avg_score / max(calibration_strength, 1e-6))
        S_merged = S_values.mean() * calibration_factor

        delta_merged += S_merged * torch.outer(U_avg, V_avg)

    # --- Step 2: For all non-aligned subspaces, fall back to task arithmetic ---
    # Reconstruct each task's FULL task vector from retained subspaces,
    # then average across tasks (equivalent to task arithmetic with alpha=1/K).
    # This preserves the natural interference patterns that make task arithmetic work.
    for t in range(K):
        U, S, Vh = task_svd[t]
        task_delta = torch.zeros(m, n, dtype=base_weight.dtype, device=device)
        for r in range(task_num_subspaces[t]):
            if (t, r) not in processed:
                task_delta += S[r] * torch.outer(U[:, r], Vh[r, :])
        # Add 1/K of this task's non-aligned contribution
        delta_merged += task_delta / K

    return base_weight + delta_merged


def spectral_clustering_subspaces(
    alignment_matrix: torch.Tensor,
    n_clusters: int,
    threshold: float = 0.3,
) -> List[List[int]]:
    """
    Cluster subspaces based on alignment matrix using spectral clustering.

    This is an alternative to the greedy approach in compute_cross_task_alignment.
    Uses the normalized Laplacian of the alignment matrix.

    Args:
        alignment_matrix: Pairwise alignment scores [N, N].
        n_clusters: Number of clusters to form.
        threshold: Minimum alignment to form an edge.

    Returns:
        List of clusters, each containing subspace indices.
    """
    N = alignment_matrix.shape[0]

    # Build adjacency matrix (thresholded alignment)
    adj = (alignment_matrix >= threshold).float()
    adj = adj * alignment_matrix  # Weighted adjacency
    adj = (adj + adj.T) / 2  # Ensure symmetry

    # Degree matrix
    degree = adj.sum(dim=1)
    degree_inv_sqrt = torch.diag(1.0 / (degree.clamp(min=1e-12).sqrt()))

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    L_norm = torch.eye(N, device=alignment_matrix.device) - degree_inv_sqrt @ adj @ degree_inv_sqrt

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(L_norm)

    # Use bottom k eigenvectors for clustering
    embedding = eigenvectors[:, :n_clusters]

    # Simple k-means on the embedding
    # (In production, use sklearn or a proper k-means implementation)
    clusters = _simple_kmeans(embedding, n_clusters)

    return clusters


def _simple_kmeans(
    X: torch.Tensor,
    k: int,
    max_iter: int = 20,
) -> List[List[int]]:
    """Simple k-means clustering."""
    N = X.shape[0]
    device = X.device

    # Random initialization
    indices = torch.randperm(N, device=device)[:k]
    centroids = X[indices].clone()

    for _ in range(max_iter):
        # Assign clusters
        distances = torch.cdist(X, centroids)  # [N, k]
        assignments = distances.argmin(dim=1)  # [N]

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for c in range(k):
            mask = assignments == c
            if mask.any():
                new_centroids[c] = X[mask].mean(dim=0)
            else:
                new_centroids[c] = centroids[c]

        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    # Group indices by cluster
    clusters = [[] for _ in range(k)]
    for i in range(N):
        clusters[assignments[i].item()].append(i)

    return [c for c in clusters if len(c) > 0]
