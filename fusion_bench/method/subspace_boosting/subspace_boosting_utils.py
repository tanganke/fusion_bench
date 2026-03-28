"""
Subspace Boosting Utilities for Model Merging

This module implements the core Subspace Boosting algorithm from:
    Skorobogat et al., "Subspace-Boosted Model Merging", arXiv:2506.16506

The key idea is to mitigate rank collapse in merged task vectors by boosting
underutilized singular values after SVD decomposition.

Algorithm (per weight matrix):
    1. Decompose the merged task vector via SVD: W = U @ diag(S) @ Vh
    2. Find cutoff index k where cumulative sum of S reaches threshold beta
    3. Clamp all singular values >= S[k] (boost smaller ones)
    4. Reconstruct: W' = U @ diag(S_clamped) @ Vh
"""

import logging
from typing import List, Optional

import torch
from torch import Tensor

log = logging.getLogger(__name__)


def subspace_boosting_single_matrix(
    param: Tensor,
    beta: float = 0.01,
) -> Tensor:
    """
    Apply Subspace Boosting to a single weight matrix.

    Args:
        param: Weight matrix to process.
        beta: Cumulative sum threshold for determining cutoff.
              Singular values whose cumulative sum ratio is < beta get boosted.
              Default is 0.01 (boost bottom ~1% of energy).

    Returns:
        Weight matrix with boosted singular values.
    """
    U, S, Vh = torch.linalg.svd(param, full_matrices=False)

    # Find cutoff index where cumulative sum reaches beta threshold
    total_sum = S.sum()
    cumulative = torch.cumsum(S, dim=0)
    ratio = cumulative / total_sum
    k = (ratio >= beta).nonzero(as_tuple=False)
    cutoff_idx = k[0].item() if len(k) > 0 else len(S) - 1

    # Clamp smaller singular values to the cutoff value
    S_clamped = torch.clamp(S, min=S[cutoff_idx])

    # Reconstruct the weight matrix
    return (U * S_clamped.unsqueeze(0)) @ Vh


def _per_qkv_subspace_boosting(
    param: Tensor,
    beta: float = 0.01,
) -> Tensor:
    """
    Apply Subspace Boosting per Q, K, V projection in attention in_proj_weight.

    For ViT attention in_proj_weight which concatenates [W_Q, W_K, W_V],
    we split them, apply subspace boosting independently, and recombine.

    Args:
        param: Attention in_proj_weight of shape (3*embed_dim, embed_dim).
        beta: Cumulative sum threshold.

    Returns:
        Processed weight matrix.
    """
    embed_dim = param.shape[1]
    W_Q, W_K, W_V = torch.split(param, embed_dim, dim=0)

    processed = []
    for W in [W_Q, W_K, W_V]:
        processed.append(subspace_boosting_single_matrix(W, beta))

    return torch.cat(processed, dim=0)


def subspace_boosting(
    merged_state_dict: dict,
    beta: float = 0.01,
    attn_beta: Optional[float] = None,
    apply_to_attn: str = "per_qkv",
    target_keys: Optional[List[str]] = None,
) -> dict:
    """
    Apply Subspace Boosting to a merged task vector state dict.

    This is the main entry point for applying Subspace Boosting to a merged
    model's task vector. It processes each relevant weight matrix independently.

    Args:
        merged_state_dict: State dict of the merged task vector (not the full model).
        beta: Cumulative sum threshold for non-attention layers.
        attn_beta: Cumulative sum threshold for attention layers.
            If None, uses the same value as beta.
        apply_to_attn: How to process attention weights.
            - "per_qkv": Process Q, K, V projections independently (recommended)
            - "full_attn": Process the full in_proj_weight as one matrix
            - "none": Skip attention weights
        target_keys: List of key substrings to process. If None, processes all
            linear/attention layers. Default targets ViT-style layers.

    Returns:
        State dict with subspace-boosted weight matrices.
    """
    if attn_beta is None:
        attn_beta = beta

    if target_keys is None:
        # Default target keys for ViT/CLIP models
        target_keys = [
            "attn.in_proj_weight",
            "attn.out_proj.weight",
            "c_fc.weight",
            "c_proj.weight",
            "model.visual.proj",
            # Generic transformer keys
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.out_proj.weight",
            "fc1.weight",
            "fc2.weight",
            # T5-style keys
            "q.weight",
            "k.weight",
            "v.weight",
            "o.weight",
            "wi.weight",
            "wi_0.weight",
            "wi_1.weight",
            "wo.weight",
        ]

    result = {}
    for key, param in merged_state_dict.items():
        if not isinstance(param, Tensor):
            result[key] = param
            continue

        # Check if this key should be processed
        should_process = False
        is_attn_in_proj = False
        current_beta = beta

        for target in target_keys:
            if target in key:
                should_process = True
                if "in_proj" in key or key.endswith("q.weight") or key.endswith("k.weight") or key.endswith("v.weight"):
                    is_attn_in_proj = True
                    current_beta = attn_beta
                break

        if not should_process:
            result[key] = param
            continue

        # Apply subspace boosting
        if is_attn_in_proj and apply_to_attn == "per_qkv":
            result[key] = _per_qkv_subspace_boosting(param, current_beta)
        else:
            result[key] = subspace_boosting_single_matrix(param, current_beta)

    return result
