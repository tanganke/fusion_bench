R"""
```python
# Get the task-wise weights
task_wise_weights = get_task_wise_weights(num_models)

# Define the task vectors (in this case, we'll use the state_dict of the pretrained model)
task_vectors = ...

# Initialize the TaskWiseMergedModel
merged_model = TaskWiseMergedModel(pretrained_model, task_wise_weights, task_vectors)

# Now you can use the merged_model like a regular PyTorch model
outputs = merged_model(inputs)
```
"""

import functools
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional  # noqa: F401

import torch
from torch import Tensor, nn
from torch.func import functional_call

from fusion_bench.models.utils import StateDictType, del_attr, get_attr, set_attr
from fusion_bench.utils.type import StateDictType, TorchModelType

log = logging.getLogger(__name__)

__all__ = ["get_task_wise_weights", "fuse_weights", "TaskWiseMergedModel"]


def get_task_wise_weights(num_models: int, init_values: float = None) -> Tensor:
    """
    This function generates a tensor of weights for each model.

    Args:
        num_models (int): The number of models.
        init_values (float, optional): The initial value for each weight. Defaults to None.

    Returns:
        Tensor: A tensor of weights for each model.
    """
    assert num_models >= 1, f"num_models must be >= 1, got {num_models}"
    if init_values is None:
        init_values = 1.0 / num_models
    return torch.full((num_models,), init_values, dtype=torch.float32)


def _fuse_weights(task_wise_weight: Tensor, tensors: List[Tensor]) -> Tensor:
    """
    This function fuses the weights of the models.

    Args:
        task_wise_weight (Tensor): The weights for each model.
        tensors (List[Tensor]): The list of tensors to be fused.

    Returns:
        Tensor: The fused weights.
    """
    device = task_wise_weight.device
    return sum(task_wise_weight[i] * w.to(device) for i, w in enumerate(tensors))


def fuse_weights(
    task_wise_weight: Tensor, state_dicts: List[StateDictType]
) -> StateDictType:
    """
    This function fuses the weights of the models and returns a state dictionary.

    Args:
        task_wise_weight (Tensor): The weights for each model. on cuda or cpu.
        state_dicts (List[StateDictType]): The list of state dictionaries. on cpu.

    Returns:
        StateDictType: The fused state dictionary.
    """
    num_models = len(state_dicts)
    assert (
        task_wise_weight.dim() == 1
    ), f"task_wise_weight must be a 1D tensor, got {task_wise_weight.dim()}"
    assert num_models == task_wise_weight.size(
        0
    ), f"num_models must be equal to the number of state_dicts, got {num_models} and {task_wise_weight.size(0)}"
    return {
        k: _fuse_weights(task_wise_weight, [sd[k] for sd in state_dicts])
        for k in state_dicts[0].keys()
    }


class TaskWiseMergedModel(nn.Module, Generic[TorchModelType]):
    """
    A PyTorch module that dynamically merges multiple fine-tuned models using learnable task-wise weights.

    This class implements a sophisticated model fusion approach where multiple task-specific models
    are combined with a pretrained base model using learnable weights. The fusion is performed
    using task vectors (differences between fine-tuned and pretrained models) that are weighted
    and added to the base model's parameters.

    The key innovation is that the merging weights are learnable parameters that can be optimized
    during training, allowing the model to automatically learn the optimal combination of different
    task-specific knowledge.

    Architecture:
        - Base pretrained model (frozen)
        - Multiple task vectors (differences from pretrained model, frozen)
        - Learnable task-wise weights (trainable parameters)
        - Dynamic merging during forward pass

    Args:
        task_wise_weight (Tensor): Initial weights for each task model. Shape: (num_models,).
            These become learnable parameters that control the contribution of each task vector.
        pretrained_model (TorchModelType): The base pretrained model that serves as the foundation.
            This model is frozen and used as the starting point for merging.
        finetuned_models (List[TorchModelType]): List of fine-tuned models for different tasks.
            These are converted to task vectors (differences from pretrained model) and frozen.
        clamp_weights (bool, optional): Whether to clamp merge weights to [0, 1] range.
            Defaults to True. When True, ensures weights are non-negative and bounded.
        tie_weights (bool, optional): Whether to tie weights during functional call.
            Defaults to False. Used in the underlying PyTorch functional_call.
        strict (bool, optional): Whether to enforce strict parameter matching.
            Defaults to True. Used in the underlying PyTorch functional_call.
        task_vector_dtype (Optional[torch.dtype], optional): Data type for task vectors.
            Defaults to None. Can be used to save memory (e.g., torch.float16).

    Attributes:
        merge_weight (nn.Parameter): Learnable weights for merging task vectors.
        pretrained_model (TorchModelType): The frozen base model.
        task_vectors (nn.ModuleList): List of frozen task vector models.
        _merged_state_dict (StateDictType): Cached merged state dictionary.

    Example:
        ```python
        import torch
        import torch.nn as nn

        # Create example models
        pretrained_model = nn.Linear(10, 5)
        finetuned_model1 = nn.Linear(10, 5)  # Fine-tuned on task 1
        finetuned_model2 = nn.Linear(10, 5)  # Fine-tuned on task 2

        # Initialize task-wise weights
        task_weights = torch.tensor([0.3, 0.7])  # Initial weights for 2 tasks

        # Create merged model
        merged_model = TaskWiseMergedModel(
            task_wise_weight=task_weights,
            pretrained_model=pretrained_model,
            finetuned_models=[finetuned_model1, finetuned_model2],
            clamp_weights=True
        )

        # Use like a regular PyTorch model
        x = torch.randn(32, 10)
        output = merged_model(x)

        # Train the merge weights
        optimizer = torch.optim.Adam(merged_model.parameters())
        loss = some_loss_function(output, targets)
        loss.backward()
        optimizer.step()

        # Get the final merged model
        final_model = merged_model.merge_and_unload()
        ```

    Training Workflow:
        1. **Initialization**: Task vectors are computed as differences from pretrained model
        2. **Forward Pass**: Weights are dynamically merged based on current merge_weight values
        3. **Loss Computation**: Standard loss computation on model outputs
        4. **Backpropagation**: Gradients flow through merge_weight parameters
        5. **Optimization**: merge_weight parameters are updated to improve performance

    Memory Efficiency:
        - Task vectors can use lower precision (task_vector_dtype)
        - Base model and task vectors are frozen (no gradient computation)
        - Only merge weights require gradients

    Note:
        - The pretrained model and task vectors are frozen during training
        - Only the merge weights (task_wise_weight) are trainable parameters
        - Task vectors represent the difference between fine-tuned and pretrained models
        - The merged state dict is cached and recomputed when merge weights change
    """

    _merged_state_dict: StateDictType = None

    def __init__(
        self,
        task_wise_weight: Tensor,
        pretrained_model: TorchModelType,
        finetuned_models: List[TorchModelType],
        clamp_weights: bool = True,
        tie_weights: bool = False,
        strict: bool = True,
        task_vector_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the TaskWiseMergedModel.

        This constructor sets up the model by:
        1. Converting fine-tuned models to task vectors (differences from pretrained)
        2. Freezing the pretrained model and task vectors
        3. Setting up learnable merge weights as parameters
        4. Configuring merging behavior options

        Args:
            task_wise_weight (Tensor): Initial weights for each task model. Shape: (num_models,).
                These values become the starting point for learnable parameters.
            pretrained_model (TorchModelType): The base pretrained model.
                Will be frozen and used as the foundation for merging.
            finetuned_models (List[TorchModelType]): List of fine-tuned models.
                Must have the same architecture as pretrained_model.
            clamp_weights (bool, optional): Whether to clamp weights to [0, 1]. Defaults to True.
            tie_weights (bool, optional): Whether to tie weights in functional_call. Defaults to False.
            strict (bool, optional): Whether to use strict parameter matching. Defaults to True.
            task_vector_dtype (Optional[torch.dtype], optional): Data type for task vectors.
                Defaults to None (same as original models).

        Raises:
            ValueError: If the number of task_wise_weights doesn't match the number of fine-tuned models.
            RuntimeError: If models have incompatible architectures.
        """
        super().__init__()
        self.clamp_weights = clamp_weights
        self.tie_weights = tie_weights
        self.strict = strict
        self.task_vector_dtype = task_vector_dtype

        self.merge_weight = nn.Parameter(task_wise_weight, requires_grad=True)

        for name, param in pretrained_model.named_parameters():
            if not param.requires_grad:
                for m in finetuned_models:
                    del_attr(m, name.split("."))
            else:
                for m in finetuned_models:
                    get_attr(m, name.split(".")).data = (
                        get_attr(m, name.split(".")) - param
                    )
        self.pretrained_model = pretrained_model.requires_grad_(False)
        for m in finetuned_models:
            m.requires_grad_(False)
        self.task_vectors = nn.ModuleList(finetuned_models)
        if self.task_vector_dtype is not None:
            self.task_vectors = self.task_vectors.to(self.task_vector_dtype)

    @property
    def forward_model(self):
        """
        Get a functional model with merged parameters.

        Returns a partial function that applies the pretrained model with the current
        merged state dictionary. This allows for efficient forward passes without
        modifying the original model's parameters.

        Returns:
            Callable: A partial function that can be called with (args, kwargs) to
                perform forward pass with merged parameters.

        Example:
            ```python
            # Internal usage during forward pass
            forward_fn = merged_model.forward_model
            output = forward_fn(args=(x,), kwargs={})
            ```
        """
        return functools.partial(
            functional_call,
            self.pretrained_model,
            self._merged_state_dict,
            tie_weights=self.tie_weights,
            strict=self.strict,
        )

    def merge_weights(self, task_vector_mask: Optional[Dict[str, Tensor]] = None):
        """
        Merge task vectors with the pretrained model using current merge weights.

        This method computes the merged model parameters by combining the pretrained
        model with weighted task vectors. The resulting state dictionary represents
        a model that incorporates knowledge from all task-specific models.

        The merging formula for each parameter is:
        merged_param = pretrained_param + Σ(weight_i * task_vector_i * mask_i)

        Args:
            task_vector_mask (Optional[Dict[str, Tensor]], optional): Optional masks
                to selectively apply task vectors to specific parameters. Keys should
                match parameter names, values should be tensors with the same shape
                as the corresponding parameters. Defaults to None (no masking).

        Returns:
            StateDictType: The merged state dictionary containing combined parameters.

        Example:
            ```python
            # Basic merging
            merged_state = model.merge_weights()

            # Merging with parameter-specific masks
            masks = {
                'layer1.weight': torch.ones_like(model.pretrained_model.layer1.weight),
                'layer2.weight': torch.zeros_like(model.pretrained_model.layer2.weight),
            }
            masked_state = model.merge_weights(task_vector_mask=masks)
            ```

        Note:
            - If clamp_weights is True, merge weights are clamped to [0, 1] range
            - The merged state dict is cached in _merged_state_dict
            - Task vector masks allow fine-grained control over which parameters are affected
        """
        if self.clamp_weights:
            merge_weight = self.merge_weight.clamp(0, 1)
        else:
            merge_weight = self.merge_weight

        state_dict = self.pretrained_model.state_dict(keep_vars=True)
        for weight, task_vector in zip(merge_weight, self.task_vectors):
            for name, param in task_vector.named_parameters():
                if task_vector_mask is None:
                    w = weight
                else:
                    w = weight * task_vector_mask[name]
                state_dict[name] = state_dict[name] + param * w
        self._merged_state_dict = state_dict
        return state_dict

    def merge_and_unload(
        self,
        task_vector_mask: Optional[Dict[str, Tensor]] = None,
        copy: bool = False,
    ) -> TorchModelType:
        """
        Merge models and return the final merged model.

        This method performs the merging operation and then loads the merged parameters
        into the pretrained model, returning a standard PyTorch model that can be used
        independently of the TaskWiseMergedModel wrapper.

        Args:
            task_vector_mask (Optional[Dict[str, Tensor]], optional): Optional masks
                for selective parameter merging. Defaults to None.
            copy (bool, optional): Whether to return a deep copy of the pretrained model.
                Defaults to False. If True, the original pretrained model remains unchanged.

        Returns:
            TorchModelType: The pretrained model with merged parameters loaded.
                This is a standalone model that can be used without the wrapper.

        Example:
            ```python
            # Train the merged model
            for epoch in range(num_epochs):
                # ... training loop ...
                pass

            # Get the final merged model
            final_model = merged_model.merge_and_unload()

            # Save or use the final model
            torch.save(final_model.state_dict(), 'merged_model.pth')
            output = final_model(new_input)
            ```

        Warning:
            This method modifies the pretrained_model's parameters in-place.
            The original pretrained model parameters will be lost.
        """
        self.merge_weights(task_vector_mask=task_vector_mask)
        if copy:
            model = deepcopy(self.pretrained_model)
        else:
            model = self.pretrained_model
        model.load_state_dict(self._merged_state_dict)
        return model

    def forward(self, *args, **kwargs):
        """
        Forward pass through the dynamically merged model.

        This method performs the forward pass by first ensuring the model parameters
        are merged according to the current merge weights, then applying the merged
        model to the input data.

        The forward pass involves:
        1. Check if merged state dict is current (recompute if needed)
        2. Apply the merged model to inputs using functional_call
        3. Return the model outputs

        Args:
            *args: Positional arguments to pass to the underlying model.
            **kwargs: Keyword arguments to pass to the underlying model.

        Returns:
            Any: The output of the merged model, typically torch.Tensor or tuple of tensors.

        Example:
            ```python
            # Single input
            x = torch.randn(32, 784)
            output = merged_model(x)

            # Multiple inputs
            x1, x2 = torch.randn(32, 784), torch.randn(32, 100)
            output = merged_model(x1, x2)

            # With keyword arguments
            output = merged_model(input_ids=input_ids, attention_mask=attention_mask)
            ```

        Note:
            - The merged state dict is recomputed if merge weights have changed
            - This allows for dynamic behavior during training as weights are updated
            - The computation is efficient as merging only happens when needed
        """
        if self._merged_state_dict is None:
            self.merge_weights()
        return self.forward_model(args=args, kwargs=kwargs)

    # def __getattr__(self, name: str) -> Any:
    #     try:
    #         return super().__getattr__(name)
    #     except AttributeError:
    #         attr = getattr(self.pretrained_model, name)
    #         if isinstance(attr, Callable):
    #             warnings.warn(
    #                 f"forwarding `{name}` to the underlying model", UserWarning
    #             )
    #         return attr

    # def __setattr__(self, name: str, value: Any) -> None:
    #     try:
    #         super().__setattr__(name, value)
    #     except AttributeError:
    #         setattr(self.pretrained_model, name, value)
