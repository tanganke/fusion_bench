"""
SpectralMerge: Spectral-Aware Model Merging

This module implements SpectralMerge, which performs model merging directly
in the spectral domain rather than applying post-hoc corrections.

Key insight: Task vectors contain both shared directions (large singular values,
high cross-task overlap) that require calibration, and task-specific directions
(small singular values, orthogonal) that can be preserved as-is.

Unlike SVC which rescales singular values after merging, SpectralMerge:
1. Decomposes each task vector via SVD
2. Detects cross-task spectral alignment BEFORE merging
3. Merges within aligned subspaces using calibrated weights
4. Preserves orthogonal task-specific directions untouched

Reference: SpectralMerge (Tang et al., 2026)
"""

import os
from typing import List, Optional

import torch
from tqdm import tqdm

from fusion_bench import BaseAlgorithm, BaseModelPool, auto_register_config

from .utils import (
    compute_cross_task_alignment,
    merge_in_spectral_domain,
    spectral_clustering_subspaces,
)


@auto_register_config
class SpectralMerging(BaseAlgorithm):
    """
    Spectral-aware model merging that operates in the spectral domain.

    Unlike post-hoc calibration methods (e.g., SVC), SpectralMerge detects
    cross-task alignment patterns before merging and applies calibrated merging
    directly in the decomposed spectral space.

    Args:
        energy_threshold: Fraction of spectral energy to retain (default: 0.95).
            Higher values retain more subspaces but may include noise.
        alignment_threshold: Minimum cross-task alignment to consider subspaces
            as "shared" (default: 0.3). Lower values merge more aggressively.
        calibration_strength: Controls how aggressively to calibrate within
            aligned subspaces (default: 1.0). Higher = more calibration.
        device: Device for computation (default: auto-detect).
    """

    def __init__(
        self,
        energy_threshold: float = 0.95,
        alignment_threshold: float = 0.3,
        calibration_strength: float = 1.0,
        elect_threshold: float = 0.05,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.energy_threshold = energy_threshold
        self.alignment_threshold = alignment_threshold
        self.calibration_strength = calibration_strength
        self.elect_threshold = elect_threshold
        self.device = device

    def _get_device(self):
        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @torch.no_grad()
    def run(self, modelpool):
        """
        Run SpectralMerge on the given model pool.

        Args:
            modelpool: Pool of models to merge. Must contain pretrained and task models.

        Returns:
            The merged model.
        """
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        assert modelpool.has_pretrained, (
            "Pretrained model not found in the model pool "
            "(with model name '_pretrained_')."
        )

        device = self._get_device()
        pretrained_model = modelpool.load_pretrained_model()
        task_model_names = [
            name for name in modelpool.model_names if name != "_pretrained_"
        ]
        task_models = [modelpool.load_model(name) for name in task_model_names]

        # Start from pretrained weights
        merged_model = modelpool.load_pretrained_model()

        for name, param in tqdm(
            tuple(merged_model.named_parameters()),
            desc="SpectralMerge",
        ):
            if param.dim() != 2:
                # Keep non-2D params (biases, LayerNorm) unchanged
                continue

            tqdm.write(f"Merging parameter: {name}, shape: {param.shape}")

            base_weight = pretrained_model.get_parameter(name).data.to(device)
            task_weights = [
                task_model.get_parameter(name).data.to(device)
                for task_model in task_models
            ]

            merged_weight = self._merge_parameter(
                base_weight=base_weight,
                task_weights=task_weights,
            )
            param.data.copy_(merged_weight.to(param.device), non_blocking=True)

        return merged_model

    def _merge_parameter(
        self,
        base_weight: torch.Tensor,
        task_weights: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Merge a single parameter matrix using spectral-domain merging.

        Steps:
        1. Compute task vectors: ΔW_i = W_i - W_base
        2. SVD decompose each task vector
        3. Detect cross-task alignment in right singular vector space
        4. Cluster aligned subspaces
        5. Merge within clusters, preserve orthogonal directions
        """
        device = base_weight.device
        original_dtype = base_weight.dtype
        compute_dtype = torch.float32

        base_weight = base_weight.to(dtype=compute_dtype)
        task_weights = [tw.to(dtype=compute_dtype) for tw in task_weights]

        # Compute task vectors
        delta_tasks = [tw - base_weight for tw in task_weights]
        K = len(delta_tasks)

        # Step 1: SVD decompose each task vector
        task_svd = []
        for dw in delta_tasks:
            U, S, Vh = torch.linalg.svd(dw, full_matrices=False)
            task_svd.append((U, S, Vh))

        # Step 2: Determine number of subspaces to retain per task (energy threshold)
        task_num_subspaces = []
        for _, S, _ in task_svd:
            cumulative_energy = torch.cumsum(S**2, dim=0) / (S**2).sum()
            num_sub = (
                torch.searchsorted(cumulative_energy, self.energy_threshold).item() + 1
            )
            task_num_subspaces.append(min(num_sub, len(S)))

        # Step 3: Compute cross-task alignment in right singular vector space
        # Alignment matrix A[t1, t2, r1, r2] = |V_{t1,r1}^T V_{t2,r2}|
        alignment_info = compute_cross_task_alignment(
            task_svd=task_svd,
            num_subspaces=task_num_subspaces,
            alignment_threshold=self.alignment_threshold,
        )

        # Step 4: Merge in spectral domain
        merged_weight = merge_in_spectral_domain(
            base_weight=base_weight,
            task_svd=task_svd,
            task_num_subspaces=task_num_subspaces,
            alignment_info=alignment_info,
            calibration_strength=self.calibration_strength,
            elect_threshold=self.elect_threshold,
        )

        return merged_weight.to(dtype=original_dtype)
