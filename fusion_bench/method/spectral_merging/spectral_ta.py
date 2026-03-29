"""
SpectralTA: Task Arithmetic + Spectral Conflict Clipping

Hybrid approach:
1. Compute task arithmetic delta (sum of task vectors)
2. SVD decompose each task vector to detect conflicting directions
3. For conflicting directions (sign disagreement across tasks), apply TIES-style
   sign elect: only keep the sign that the majority agrees on
4. For non-conflicting directions, keep the task arithmetic sum as-is

This preserves the information-rich task arithmetic baseline while using
spectral analysis to identify and resolve the worst interference patterns.
"""

import os
from typing import List, Optional

import torch
from tqdm import tqdm

from fusion_bench import BaseAlgorithm, BaseModelPool, auto_register_config


@auto_register_config
class SpectralTA(BaseAlgorithm):
    """
    Task Arithmetic with Spectral Conflict Resolution.

    Uses SVD to identify conflicting directions across task vectors, then
    applies sign-based elect (like TIES) only to those directions.

    Args:
        energy_threshold: Fraction of spectral energy to analyze for conflicts
            (default: 0.95). Higher = more directions checked.
        conflict_threshold: Minimum sign disagreement ratio to consider a
            direction as "conflicting" (default: 0.3). Lower = more aggressive.
        clip_mode: How to handle conflicting directions.
            'sign_elect' = majority sign vote (like TIES),
            'zero' = zero out conflicting components,
            'scale' = scale down by agreement ratio.
        device: Device for computation.
    """

    def __init__(
        self,
        energy_threshold: float = 0.95,
        conflict_threshold: float = 0.3,
        clip_mode: str = "sign_elect",
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.energy_threshold = energy_threshold
        self.conflict_threshold = conflict_threshold
        self.clip_mode = clip_mode
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
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        assert modelpool.has_pretrained

        device = self._get_device()
        pretrained_model = modelpool.load_pretrained_model()
        task_model_names = [
            name for name in modelpool.model_names if name != "_pretrained_"
        ]
        task_models = [modelpool.load_model(name) for name in task_model_names]

        merged_model = modelpool.load_pretrained_model()

        for name, param in tqdm(
            tuple(merged_model.named_parameters()),
            desc="SpectralTA",
        ):
            if param.dim() != 2:
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
        device = base_weight.device
        original_dtype = base_weight.dtype
        compute_dtype = torch.float32

        base_weight = base_weight.to(dtype=compute_dtype)
        task_weights = [tw.to(dtype=compute_dtype) for tw in task_weights]

        # Step 1: Compute task vectors and their sum (standard task arithmetic)
        delta_tasks = [tw - base_weight for tw in task_weights]
        K = len(delta_tasks)

        # Standard task arithmetic sum
        delta_ta = torch.stack(delta_tasks, dim=0).sum(dim=0)

        # Step 2: SVD each task vector, detect conflicting directions
        task_svd = []
        for dw in delta_tasks:
            U, S, Vh = torch.linalg.svd(dw, full_matrices=False)
            task_svd.append((U, S, Vh))

        # Step 3: For each significant direction, check sign agreement
        # Collect all right singular vectors and their signed singular values
        num_subspaces = []
        for _, S, _ in task_svd:
            cum_energy = torch.cumsum(S**2, dim=0) / (S**2).sum()
            num_sub = (
                torch.searchsorted(cum_energy, self.energy_threshold).item() + 1
            )
            num_subspaces.append(min(num_sub, len(S)))

        # Build conflict map: for each pair of tasks, find directions where
        # right singular vectors overlap significantly but have opposite signs
        m, n = base_weight.shape
        conflict_mask = torch.zeros(m, n, dtype=torch.bool, device=device)

        # For each task pair, compute alignment and detect sign conflicts
        for t1 in range(K):
            U1, S1, Vh1 = task_svd[t1]
            for t2 in range(t1 + 1, K):
                U2, S2, Vh2 = task_svd[t2]

                # Compute pairwise alignment: |V1^T V2|
                V1 = Vh1[: num_subspaces[t1]].T  # [n, r1]
                V2 = Vh2[: num_subspaces[t2]].T  # [n, r2]
                V1_n = torch.nn.functional.normalize(V1.float(), dim=0)
                V2_n = torch.nn.functional.normalize(V2.float(), dim=0)
                alignment = V1_n.T @ V2_n  # [r1, r2]

                # Find highly aligned pairs
                high_align = (alignment.abs() > 0.5).nonzero(as_tuple=False)

                for idx in high_align:
                    r1, r2 = idx[0].item(), idx[1].item()
                    # Check sign conflict: aligned but opposite sign
                    if alignment[r1, r2] < -0.3:
                        # This direction has sign conflict between t1 and t2
                        # Mark the outer product region as conflicting
                        conflict_mask += (
                            torch.outer(U1[:, r1], Vh1[r1, :]).abs() > 0
                        )

        # Step 4: Apply clip_mode to conflicting directions
        if self.clip_mode == "sign_elect":
            # For conflicting directions, use TIES-style sign elect
            # Project delta_ta onto conflicting subspace, then resolve signs
            delta_clipped = delta_ta.clone()

            # Simple approach: for conflicting entries, take the sign of
            # the task vector with the largest magnitude
            if conflict_mask.any():
                # Stack all task deltas
                stacked = torch.stack(delta_tasks, dim=0)  # [K, m, n]
                # For conflicting positions, find the task with max |value|
                abs_stacked = stacked.abs()
                max_task_idx = abs_stacked.argmax(dim=0)  # [m, n]
                max_val = stacked.gather(0, max_task_idx.unsqueeze(0)).squeeze(0)

                # Only replace conflicting positions
                delta_clipped = torch.where(
                    conflict_mask,
                    max_val / K,  # Scale by 1/K to match TA magnitude
                    delta_ta,
                )
            return (base_weight + delta_clipped).to(dtype=original_dtype)

        elif self.clip_mode == "zero":
            # Zero out conflicting components
            delta_clipped = delta_ta.clone()
            delta_clipped[conflict_mask] = 0
            return (base_weight + delta_clipped).to(dtype=original_dtype)

        elif self.clip_mode == "scale":
            # Scale down conflicting components
            delta_clipped = delta_ta.clone()
            # Compute agreement ratio for each position
            signs = torch.stack([torch.sign(d) for d in delta_tasks], dim=0)
            agreement = signs.mean(dim=0).abs()  # [m, n], 1=all agree, 0=max conflict
            scale = torch.where(
                agreement < (1 - self.conflict_threshold),
                agreement,  # Scale by agreement ratio
                torch.ones_like(agreement),
            )
            delta_clipped = delta_ta * scale
            return (base_weight + delta_clipped).to(dtype=original_dtype)

        # Default: plain task arithmetic
        return (base_weight + delta_ta).to(dtype=original_dtype)
