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
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional  # noqa: F401

import torch
from torch import Tensor, nn
from torch.func import functional_call

from fusion_bench.utils.type import StateDictType, TorchModelType

log = logging.getLogger(__name__)

__all__ = ["get_task_wise_weights", "fuse_weights", "TaskWiseMergedModel"]


def del_attr(obj, names: List[str]):
    """
    Deletes an attribute from an object recursively.

    Args:
        obj (object): Object to delete attribute from.
        names (list): List of attribute names to delete recursively.
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names: List[str], val):
    """
    Sets an attribute of an object recursively.

    Args:
        obj (object): Object to set attribute of.
        names (list): List of attribute names to set recursively.
        val (object): Value to set the attribute to.
    """
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def get_attr(obj, names: List[str]):
    """
    Gets an attribute of an object recursively.

    Args:
        obj (object): Object to get attribute of.
        names (list): List of attribute names to get recursively.

    Returns:
        object: The attribute of the object.
    """
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])


def check_parameterNamesMatch(checkpoints: List[StateDictType]) -> None:
    """
    Checks that the parameter names of the given checkpoints match.

    Args:
        checkpoints (List[Dict[str, float]]): A list of checkpoints, where each checkpoint is a dictionary of parameter names and their corresponding values.

    Raises:
        ValueError: If the number of checkpoints is less than 2 or if the parameter names of any two checkpoints differ.

    """
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )


def get_task_wise_weights(num_models: int, init_values: float = None):
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


def _fuse_weights(task_wise_weight: Tensor, tensors: List[Tensor]):
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
        return functools.partial(
            functional_call,
            self.pretrained_model,
            self._merged_state_dict,
            tie_weights=self.tie_weights,
            strict=self.strict,
        )

    def merge_weights(self, task_vector_mask: Optional[Dict[str, Tensor]] = None):
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

    def merge_and_unload(self, task_vector_mask: Optional[Dict[str, Tensor]] = None):
        self.merge_weights(task_vector_mask=task_vector_mask)
        self.pretrained_model.load_state_dict(self._merged_state_dict)
        return self.pretrained_model

    def forward(self, *args, **kwargs):
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
