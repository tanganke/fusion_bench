"""
Subspace Boosting Algorithm for Model Merging

This module implements the Subspace Boosting method from:
    Skorobogat et al., "Subspace-Boosted Model Merging", arXiv:2506.16506

Subspace Boosting mitigates rank collapse in merged task vectors by boosting
underutilized singular values. It can be applied as a post-processing step
on top of any Task Arithmetic-based merging method (TA, TIES, Consensus).

Key insight: As more experts are merged, task vectors suffer from rank collapse
where common information dominates task-specific information. Subspace Boosting
recovers the suppressed task-specific directions by clamping small singular values
to a threshold determined by the cumulative energy.

Example usage::

    fusion_bench \\
        method=subspace_boosting \\
        method.base_method=task_arithmetic \\
        method.beta=0.01 \\
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \\
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8

References:
    - arXiv:2506.16506
    - https://github.com/ronskoro/Subspace-Boosting
"""

import logging
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from torch import Tensor, nn

from fusion_bench import LazyStateDict
from fusion_bench.compat.modelpool import to_modelpool
from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin, auto_register_config
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
)
from fusion_bench.utils.type import StateDictType

from .subspace_boosting_utils import subspace_boosting, subspace_boosting_single_matrix

log = logging.getLogger(__name__)


@auto_register_config
class SubspaceBoostingAlgorithm(SimpleProfilerMixin, BaseAlgorithm):
    """
    Subspace Boosting Algorithm for Model Merging.

    This algorithm applies Subspace Boosting as a post-processing step after
    merging task vectors using a base method (Task Arithmetic, TIES, or Consensus).

    The algorithm:
    1. Computes task vectors (finetuned - pretrained) for each model
    2. Merges task vectors using the specified base method
    3. Applies Subspace Boosting to the merged task vector:
       - SVD decomposition of each weight matrix
       - Boost underutilized singular values (clamp to threshold)
       - Reconstruct weight matrix
    4. Add boosted task vector to pretrained model

    Args:
        scaling_factor: Scaling factor for the merged task vector.
        beta: Cumulative sum threshold for boosting. Determines the cutoff
            point where singular values get clamped. Lower values boost more
            aggressively. Default: 0.01.
        attn_beta: Threshold for attention layers. If None, uses beta.
        base_method: Base merging method. Options: "task_arithmetic", "ties".
        apply_to_attn: How to handle attention in_proj weights.
            "per_qkv" (recommended), "full_attn", or "none".
        ties_threshold: Threshold for TIES merging (if base_method="ties").
        ties_merge_func: Merge function for TIES ("sum", "mean", "max").
        remove_keys: Keys to exclude from processing.
    """

    def __init__(
        self,
        scaling_factor: float = 1.0,
        beta: float = 0.01,
        attn_beta: Optional[float] = None,
        base_method: Literal["task_arithmetic", "ties"] = "task_arithmetic",
        apply_to_attn: Literal["per_qkv", "full_attn", "none"] = "per_qkv",
        ties_threshold: float = 20,
        ties_merge_func: Literal["sum", "mean", "max"] = "sum",
        remove_keys: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.scaling_factor = scaling_factor
        self.beta = beta
        self.attn_beta = attn_beta if attn_beta is not None else beta
        self.base_method = base_method
        self.apply_to_attn = apply_to_attn
        self.ties_threshold = ties_threshold
        self.ties_merge_func = ties_merge_func
        self.remove_keys = remove_keys or []

    @torch.no_grad()
    def run(
        self, modelpool: BaseModelPool | Dict[str, nn.Module], **kwargs: Any
    ) -> nn.Module:
        """
        Run the Subspace Boosting algorithm.

        Args:
            modelpool: The model pool containing pretrained and finetuned models.

        Returns:
            The merged model with subspace-boosted task vectors.
        """
        log.info("Fusing models using Subspace Boosting.")
        modelpool = to_modelpool(modelpool)

        with self.profile("loading models"):
            pretrained_model = modelpool.load_model("_pretrained_")
            ft_checks: List[StateDictType] = [
                modelpool.load_model(model_name).state_dict()
                for model_name in modelpool.model_names
            ]
            ptm_check: StateDictType = pretrained_model.state_dict()

        with self.profile("computing task vectors"):
            # Compute task vectors: tau_i = theta_i - theta_base
            task_vectors: List[StateDictType] = [
                state_dict_sub(ft, ptm_check) for ft in ft_checks
            ]

        with self.profile("base merging"):
            # Merge task vectors using the base method
            if self.base_method == "task_arithmetic":
                merged_tv = self._task_arithmetic_merge(task_vectors)
            elif self.base_method == "ties":
                merged_tv = self._ties_merge(task_vectors, ptm_check)
            else:
                raise ValueError(f"Unknown base method: {self.base_method}")

        with self.profile("subspace boosting"):
            # Apply Subspace Boosting to the merged task vector
            merged_tv = subspace_boosting(
                merged_tv,
                beta=self.beta,
                attn_beta=self.attn_beta,
                apply_to_attn=self.apply_to_attn,
                target_keys=None,  # Use default targets
            )

        with self.profile("reconstructing model"):
            # Apply scaling and add to pretrained: theta_m = theta_base + lambda * boosted_tv
            merged_tv = state_dict_mul(merged_tv, self.scaling_factor)
            merged_state_dict = state_dict_add(ptm_check, merged_tv)

        with self.profile("loading state dict"):
            # Load the merged state dict into the model
            if isinstance(pretrained_model, nn.Module):
                model = pretrained_model
                model.load_state_dict(merged_state_dict)
            elif isinstance(pretrained_model, LazyStateDict):
                model = deepcopy(pretrained_model.meta_module)
                model = model.to_empty(device=pretrained_model._device)
                result = model.load_state_dict(merged_state_dict, strict=False)
                if result.unexpected_keys:
                    raise ValueError(
                        f"Unexpected keys in state dict: {result.unexpected_keys}"
                    )
                if result.missing_keys:
                    log.warning(f"Missing keys in state dict: {result.missing_keys}")
            else:
                raise TypeError(f"Unsupported model type: {type(pretrained_model)}")

        self.print_profile_summary()
        return model

    def _task_arithmetic_merge(
        self, task_vectors: List[StateDictType]
    ) -> StateDictType:
        """Merge task vectors using simple averaging (Task Arithmetic)."""
        merged = None
        for tv in task_vectors:
            if merged is None:
                merged = deepcopy(tv)
            else:
                merged = state_dict_add(merged, tv)
        return merged

    def _ties_merge(
        self, task_vectors: List[StateDictType], ptm_check: StateDictType
    ) -> StateDictType:
        """
        Merge task vectors using TIES merging.

        TIES: Trim, Elect sign, then Disjoint mean.
        """
        from fusion_bench.method.ties_merging.ties_merging_utils import (
            state_dict_to_vector,
            ties_merging,
            vector_to_state_dict,
        )

        # Convert to flat vectors for TIES processing
        flat_tvs = torch.vstack(
            [state_dict_to_vector(tv, self.remove_keys) for tv in task_vectors]
        )

        # Apply TIES merging
        merged_flat = ties_merging(
            flat_tvs,
            reset_thresh=self.ties_threshold,
            merge_func=self.ties_merge_func,
        )

        # Convert back to state dict
        # We need a reference state dict, use the first task vector
        return vector_to_state_dict(merged_flat, task_vectors[0], self.remove_keys)
