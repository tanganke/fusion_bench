R"""
# Task Singular Vector Merging (TSVM) Algorithm Implementation

This module implements the Task Singular Vector Merging algorithm for combining multiple fine-tuned models
into a single unified model.

## Example Usage:

Merge 8 CLIP-ViT-B/32 models with TSVM:

```bash
fusion_bench \
    method=task_singular_vector/TaskSingularVectorMerging \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

Merge 8 CLIP-ViT-B/32 models with TSVM and return individual transformed models:

```bash
fusion_bench \
    method=task_singular_vector/TaskSingularVectorMerging \
    method.return_single_task_models=true \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

Merge 20 CLIP-VIT-B/32 models with TSVM:

```bash
fusion_bench \
    method=task_singular_vector/TaskSingularVectorMerging \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20
```

## References:

- Gargiulo, et al. Task Singular Vectors: Reducing Task Interference in Model Merging. 
    https://arxiv.org/abs/2412.00081
- See `docs/algorithms/task_singular_vector.md` for more details.
"""

from copy import deepcopy
from typing import Iterable, List, Optional, Union

import torch
from omegaconf import ListConfig
from torch import Tensor, nn

import fusion_bench as fb
from fusion_bench import BaseAlgorithm
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.utils import timeit_context
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
)
from fusion_bench.utils.type import StateDictType

from .utils import (
    TSVM_utils,
    check_parameterNamesMatch,
    check_state_dicts_equal,
    state_dict_to_vector,
    vector_to_state_dict,
)


class TaskSingularVectorMerging(BaseAlgorithm, LightningFabricMixin):
    """
    Task Singular Vector Merging (TSVM) Algorithm

    This class implements a model merging technique that leverages Singular Value
    Decomposition (SVD) to identify and combine the most important directions in the task vector
    space. The algorithm is particularly effective for merging multiple models fine-tuned on
    different tasks while preserving their essential capabilities.

    Key Concepts:
    - Task Vector: The difference between a fine-tuned model and its pretrained base model,
      representing the knowledge gained during fine-tuning for a specific task.
    - Singular Value Decomposition: A matrix factorization technique used to find the principal
      components (most important directions) in the space of task vectors.
    - Model Merging: The process of combining multiple models into a single unified model that
      retains capabilities from all constituent models.

    Algorithm Steps:
    1. Extract task vectors from all fine-tuned models by subtracting the pretrained model
    2. Apply SVD to the matrix of task vectors to find principal components
    3. Reconstruct task vectors using only the most significant singular vectors
    4. Merge the reconstructed task vectors (either individually scaled or as a sum)
    5. Add the final merged task vector to the pretrained model to create the unified model

    see `docs/algorithms/task_singular_vector.md` for comprehensive algorithmic details.
    """

    def __init__(
        self,
        alpha: Optional[Union[float, Iterable[float]]] = None,
        exclude_keys: Optional[List[str]] = None,
        return_single_task_models: bool = False,
        **kwargs,
    ):
        """
        Initialize the Task Singular Vector Merging algorithm.

        Args:
            alpha (Union[float, Iterable[float]], optional): Scaling factor(s) for task vectors.
                This parameter controls the strength of the task-specific adaptations in the final model.

                - If a single float: Applied to the final merged task vector after SVD reconstruction.
                  This uniformly scales the entire merged adaptation.

                - If an iterable of floats: Applied to individual task vectors before SVD and merging.
                  Must have the same length as the number of models in the modelpool.
                  This allows for task-specific weighting (e.g., giving more importance to certain tasks).

                - If None: No scaling is applied (equivalent to alpha=1.0).

                Example: alpha=[0.8, 1.2, 0.5] would apply different weights to three different task vectors.

            exclude_keys (Optional[List[str]], optional): List of parameter names to exclude from TSVM.
                These parameters will not participate in the SVD computation and merging process.
                Useful for excluding certain layers (e.g., task-specific heads, normalization layers)
                that should not be merged across tasks. Defaults to an empty list.

                Example: exclude_keys=['classifier.weight', 'classifier.bias'] to skip classification heads.

            return_single_task_models (bool, optional): Whether to return individual transformed models.

                - If True: Returns a dictionary containing both individual models with their transformed
                  task vectors applied AND the final merged model. The dictionary has the structure:

                  >>> {'model_name_1': transformed_model_1, ..., 'merged': final_merged_model}

                - If False: Returns only the final merged model.

                This is useful for analysis or when you need access to intermediate results.
                Defaults to False.

            **kwargs: Additional arguments passed to the parent BaseAlgorithm class.

        Note:
            The choice between single alpha vs. list of alphas affects the merging strategy:
            - Single alpha: SVD is applied first, then the result is scaled
            - List of alphas: Individual task vectors are scaled first, then SVD is applied
        """
        self.alpha = alpha
        self.exclude_keys = exclude_keys if exclude_keys is not None else []
        self.return_single_task_models = return_single_task_models
        super().__init__(**kwargs)

    def load_pretrained_model_and_task_vectors(self, modelpool: fb.BaseModelPool):
        """
        Load the pretrained base model and compute task vectors from all fine-tuned models.

        This method performs the initial step of the TSVM algorithm by:
        1. Loading the original pretrained model (before any task-specific fine-tuning)
        2. For each fine-tuned model in the pool:
           - Load the fine-tuned model
           - Compute the task vector (fine-tuned params - pretrained params)
           - Optionally apply individual scaling if alpha is provided as a list

        Task vectors represent the knowledge gained during fine-tuning and are the core
        data structure that TSVM operates on.

        Args:
            modelpool (fb.BaseModelPool): Pool containing the pretrained model and all
                fine-tuned models to be merged.

        Returns:
            tuple: A tuple containing:
                - pretrained_model: The original pretrained model (torch.nn.Module)
                - task_vectors: List of task vectors (List[StateDictType]), where each
                  task vector is a state dictionary representing the parameter differences
                  for one specific task
        """
        # Load the original pretrained model that serves as the base for all fine-tuned variants
        pretrained_model = modelpool.load_pretrained_model()

        # Initialize list to store computed task vectors
        task_vectors = []

        # Process each fine-tuned model in the modelpool
        for model_idx, model_name in enumerate(modelpool.model_names):
            # Load the current fine-tuned model
            finetuned_model = modelpool.load_model(model_name)

            # Compute task vector: difference between fine-tuned and pretrained parameters
            # This captures the task-specific adaptations learned during fine-tuning
            task_vector = state_dict_sub(
                finetuned_model.state_dict(), pretrained_model.state_dict()
            )
            task_vectors.append(task_vector)

            # Apply individual scaling to task vectors if alpha is provided as a list
            # This allows for task-specific weighting before the SVD computation
            if self.alpha is not None and isinstance(self.alpha, Iterable):
                # Ensure the number of alpha values matches the number of models
                assert len(self.alpha) == len(
                    modelpool.model_names
                ), f"Alpha list length ({len(self.alpha)}) must match number of models ({len(modelpool.model_names)})"

                # Scale the current task vector by its corresponding alpha value
                task_vectors[-1] = state_dict_mul(
                    state_dict=task_vectors[-1], scalar=self.alpha[model_idx]
                )

        return pretrained_model, task_vectors

    def run(self, modelpool: fb.BaseModelPool):
        """
        Execute the complete Task Singular Vector Merging algorithm.

        This is the main entry point that orchestrates the entire TSVM process:

        The algorithm leverages the mathematical insight that task vectors often lie in a
        lower-dimensional subspace, and SVD helps identify the most important directions
        in this subspace while filtering out noise and interference.

        Args:
            modelpool (fb.BaseModelPool): Pool of models to merge, including:
                - The pretrained base model
                - Multiple fine-tuned models (one per task)
                All models must have compatible architectures.

        Returns:
            Union[torch.nn.Module, Dict[str, torch.nn.Module]]:
                - If return_single_task_models=False: Returns the merged model
                - If return_single_task_models=True: Returns a dictionary with:
                  * Individual transformed models keyed by their original names
                  * Final merged model under the key 'merged'

        Raises:
            AssertionError: If alpha list length doesn't match the number of models
        """
        # Determine the compute device for SVD operations (GPU if available for faster computation)
        accelerator = self.fabric.device

        # Phase 1: Load pretrained model and compute task vectors from all fine-tuned models
        pretrained_model, task_vectors = self.load_pretrained_model_and_task_vectors(
            modelpool
        )

        # Phase 2: Apply SVD-based merging to the task vectors
        # This is the core of the TSVM algorithm where:
        # - Task vectors are organized into a matrix
        # - SVD finds the principal components (most important directions)
        # - Task vectors are reconstructed using only the most significant components
        # - The reconstructed vectors are merged (summed) to create a unified task vector
        new_merged_tv = TSVM_utils.compute_and_sum_svd_mem_reduction(
            task_vectors,
            exclude_keys=self.exclude_keys,  # Skip certain parameters from SVD
            accelerator=accelerator,  # Use GPU if available
            return_single_task_models=self.return_single_task_models,
        )

        # Handle the case where individual transformed task vectors are also returned
        if self.return_single_task_models:
            new_merged_tv, single_task_models = new_merged_tv

        # Phase 3: Apply global scaling to the merged task vector (if alpha is a single value)
        # This is different from individual scaling applied earlier - here we scale the
        # final merged result, which affects the overall strength of all merged adaptations
        if self.alpha is not None and isinstance(self.alpha, (float, int)):
            print(f"Scaling new merged task vector by alpha: {self.alpha}")
            new_merged_tv = state_dict_mul(state_dict=new_merged_tv, scalar=self.alpha)

        # Phase 4: Prepare individual transformed models if requested
        if self.return_single_task_models:
            models = {}
            # Create individual models by adding each transformed task vector to the pretrained base
            for model_idx, model_name in enumerate(modelpool.model_names):
                # Create a deep copy to avoid modifying the original pretrained model
                model = deepcopy(pretrained_model)
                # Apply the transformed task vector to get the individual model
                model.load_state_dict(
                    state_dict_add(model.state_dict(), single_task_models[model_idx])
                )
                models[model_name] = model

        # Phase 5: Create the final merged model by adding the merged task vector to pretrained model
        # This produces a single model that combines capabilities from all input models
        pretrained_model.load_state_dict(
            state_dict_add(new_merged_tv, pretrained_model.state_dict())
        )

        # Phase 6: Return results based on the requested output format
        if self.return_single_task_models:
            # Include the final merged model in the dictionary of results
            models["merged"] = pretrained_model
            return models
        else:
            # Return only the merged model
            return pretrained_model
