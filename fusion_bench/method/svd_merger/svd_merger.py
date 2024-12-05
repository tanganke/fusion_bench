import logging
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from fusion_bench.method.base_algorithm import BaseAlgorithm
from fusion_bench.modelpool import BaseModelPool

log = logging.getLogger(__name__)


class SVDMergerAlgorithm(BaseAlgorithm):
    """
    SVD-based Model Merging Algorithm.

    This algorithm merges multiple fine-tuned models using SVD decomposition and subspace analysis.
    It splits the parameter space into three subspaces:
    1. Most significant dimensions (default: 50% of total variance)
    2. Remaining significant dimensions
    3. Unused dimensions

    The algorithm then combines updates with orthogonalization to preserve task-specific features.
    """

    def __init__(
        self,
        task_weights: Optional[List[float]] = None,
        subspace_threshold: float = 0.5,
        rank: Optional[int] = None,
        **kwargs,
    ):
        self.task_weights = task_weights
        self.subspace_threshold = subspace_threshold
        self.rank = rank
        super().__init__(**kwargs)

    def svd_decomposition(
        self, weight_matrix: torch.Tensor, k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform SVD decomposition with optional rank reduction"""
        U, S, V = torch.svd(weight_matrix)
        if k is not None:
            U = U[:, :k]
            S = S[:k]
            V = V[:, :k]
        return U, S, V

    def find_subspaces(
        self, singular_values: torch.Tensor
    ) -> Tuple[List[int], List[int], List[int]]:
        """Split parameter space into three subspaces based on singular values"""
        total_sum = singular_values.sum()
        cumsum = 0
        space_1 = []  # Most significant dimensions
        space_2 = []  # Remaining significant
        space_3 = []  # Unused dimensions

        for i, s in enumerate(singular_values):
            cumsum += s
            if cumsum <= total_sum * self.subspace_threshold:
                space_1.append(i)
            elif s > 0:
                space_2.append(i)
            else:
                space_3.append(i)

        return space_1, space_2, space_3

    def project_orthogonal(
        self, update_1: torch.Tensor, update_2: torch.Tensor
    ) -> torch.Tensor:
        """Project update_1 to space orthogonal to update_2"""
        u1 = update_1.flatten()
        u2 = update_2.flatten()
        proj = (u1 @ u2) / (u2 @ u2) * u2
        orthogonal = u1 - proj
        return orthogonal.reshape(update_1.shape)

    def compute_weight_update(
        self, pretrained_model: nn.Module, finetuned_model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """Compute weight updates between finetuned and pretrained model"""
        updates = {}
        for name, pretrained_param in pretrained_model.named_parameters():
            finetuned_param = dict(finetuned_model.named_parameters())[name]
            updates[name] = finetuned_param.data - pretrained_param.data
        return updates

    def run(self, modelpool: BaseModelPool) -> nn.Module:
        """
        Run the SVD-based model merging algorithm.

        Args:
            modelpool: Pool of models to merge, including the base model and fine-tuned models

        Returns:
            Merged model combining features from all input models
        """
        # Get models from the pool
        base_model = modelpool.load_pretrained_model()
        finetuned_models = [
            modelpool.load_model(model) for model in modelpool.model_names
        ]

        # Set up task weights if not provided
        if self.task_weights is None:
            self.task_weights = [1.0 / len(finetuned_models)] * len(finetuned_models)

        # Create a copy of base model for merging
        merged_model = deepcopy(base_model)

        # Compute updates for each finetuned model
        all_updates = [
            self.compute_weight_update(base_model, model) for model in finetuned_models
        ]

        # Process each parameter
        for param_name, base_param in base_model.named_parameters():
            if not param_name.endswith(".weight"):
                # Simple averaging for bias terms
                merged_bias = sum(
                    self.task_weights[i] * updates[param_name]
                    for i, updates in enumerate(all_updates)
                )
                merged_model.get_parameter(param_name).data += merged_bias
                continue

            # Handle weight matrices
            weight_updates = [updates[param_name] for updates in all_updates]

            # Perform SVD on each update
            decompositions = [
                self.svd_decomposition(update, self.rank) for update in weight_updates
            ]

            # Find subspaces based on first model
            spaces = self.find_subspaces(decompositions[0][1])

            # Combine updates with orthogonalization
            merged_update = torch.zeros_like(base_param.data)
            for i, (U, S, V) in enumerate(decompositions):
                # Focus on Space II & III for task-specific updates
                for space_idx in spaces[1] + spaces[2]:
                    if space_idx < len(S):
                        update = (
                            U[:, space_idx : space_idx + 1]
                            @ torch.diag(S[space_idx : space_idx + 1])
                            @ V[:, space_idx : space_idx + 1].T
                        )

                        # Project to orthogonal space if needed
                        if i > 0:
                            update = self.project_orthogonal(update, merged_update)

                        merged_update += self.task_weights[i] * update

            # Apply merged update
            merged_model.get_parameter(param_name).data += merged_update

        return merged_model
