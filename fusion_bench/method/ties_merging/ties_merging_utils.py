"""
This is modified based on https://github.com/EnnengYang/AdaMerging/blob/main/src/ties_merging_utils.py
"""

import copy
from collections import OrderedDict
from typing import List

import torch
from torch import Tensor, nn

from fusion_bench.utils.type import StateDictType


# Model conversion utils
def state_dict_to_vector(state_dict, remove_keys=[]):
    """
    Convert a state dictionary to a vector, removing specified keys.

    Args:
        state_dict (dict): The state dictionary to convert.
        remove_keys (list): List of keys to remove from the state dictionary.

    Returns:
        Tensor: A vector representation of the state dictionary.
    """
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    """
    Convert a vector back to a state dictionary, removing specified keys.

    Args:
        vector (Tensor): The vector to convert.
        state_dict (dict): The reference state dictionary.
        remove_keys (list): List of keys to remove from the state dictionary.

    Returns:
        dict: A state dictionary representation of the vector.
    """
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the reference dict
    nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict


def add_ptm_to_tv(tv_dict, ptm_dict):
    """
    Add the values of one state dictionary to another.

    Args:
        tv_dict (dict): The target state dictionary.
        ptm_dict (dict): The state dictionary to add.

    Returns:
        dict: The resulting state dictionary after addition.
    """
    assert set(tv_dict.keys()) == set(
        ptm_dict.keys()
    ), "Differing parameter names in models."
    final_dict = copy.deepcopy(tv_dict)
    for k, v in ptm_dict.items():
        final_dict[k] = tv_dict[k] + v
    return final_dict


def check_parameterNamesMatch(checkpoints: List[StateDictType]) -> None:
    """
    Check if the parameter names match across multiple checkpoints.

    Args:
        checkpoints (list): List of state dictionaries to check.

    Raises:
        ValueError: If the parameter names do not match.
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


def check_state_dicts_equal(
    state_dict1: StateDictType, state_dict2: StateDictType
) -> bool:
    """
    Check if two state dictionaries are equal.

    Args:
        state_dict1 (dict): The first state dictionary.
        state_dict2 (dict): The second state dictionary.

    Returns:
        bool: True if the state dictionaries are equal, False otherwise.
    """
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True


# TIES MERGING UTILS


def topk_values_mask(M, K=0.7, return_mask=False):
    """
    Mask the top K values in a tensor.

    Args:
        M (Tensor): The input tensor.
        K (float): The proportion of top values to keep.
        return_mask (bool): Whether to return the mask tensor.

    Returns:
        tuple: The masked tensor, the mean of the mask, and optionally the mask tensor.
    """
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)


def resolve_zero_signs(sign_to_mult, method="majority"):
    """
    Resolve zero signs in a tensor by majority or minority rule.

    Args:
        sign_to_mult (Tensor): The tensor with signs to resolve.
        method (str): The method to use for resolving zero signs ("majority" or "minority").

    Returns:
        Tensor: The tensor with resolved signs.
    """
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


def resolve_sign(v: Tensor):
    """
    Resolve the sign of a tensor by majority rule.

    Args:
        v (Tensor): The input tensor.

    Returns:
        Tensor: The tensor with resolved signs.
    """
    sign_to_mult = torch.sign(v.sum(dim=0))
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult


def disjoint_merge(v: Tensor, merge_func: str, sign_to_mult):
    """
    Perform disjoint merging of a tensor using a specified merge function.

    Args:
        v (Tensor): The input tensor.
        merge_func (str): The merge function to use ("mean", "sum", or "max").
        sign_to_mult (Tensor): The tensor with signs to use for merging.

    Returns:
        Tensor: The merged tensor.
    """
    merge_func = merge_func.split("-")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(sign_to_mult.unsqueeze(0) > 0, v > 0, v < 0)
        selected_entries = v * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = v != 0
        selected_entries = v * rows_to_keep

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs


def ties_merging(
    flat_task_checks,
    reset_thresh=None,
    merge_func="",
):
    """
    Perform TIES merging on a tensor.

    Args:
        flat_task_checks (Tensor): The input tensor.
        reset_thresh (float): The threshold for resetting values.
        merge_func (str): The merge function to use.

    Returns:
        Tensor: The merged tensor.
    """
    all_checks = flat_task_checks.clone()
    updated_checks, *_ = topk_values_mask(all_checks, K=reset_thresh, return_mask=False)
    print("RESOLVING SIGN")
    final_signs = resolve_sign(updated_checks)
    assert final_signs is not None

    print(f"Disjoint AGGREGATION: {merge_func}")
    merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)

    return merged_tv


def disjoint_merge_split(v: Tensor, merge_func: str, sign_to_mult):
    """
    Perform disjoint merging of a tensor using a specified merge function and return selected entries.

    Args:
        v (Tensor): The input tensor.
        merge_func (str): The merge function to use ("sum").
        sign_to_mult (Tensor): The tensor with signs to use for merging.

    Returns:
        tuple: The selected entries and the merged tensor.
    """
    merge_func = merge_func.split("-")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(sign_to_mult.unsqueeze(0) > 0, v > 0, v < 0)
        selected_entries = v * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = v != 0
        selected_entries = v * rows_to_keep

    if merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return selected_entries, disjoint_aggs


def ties_merging_split(
    flat_task_checks,
    reset_thresh=None,
    merge_func: str = "",
):
    """
    Perform TIES merging on a tensor and return selected entries.

    Args:
        flat_task_checks (Tensor): The input tensor.
        reset_thresh (float): The threshold for resetting values.
        merge_func (str): The merge function to use.

    Returns:
        tuple: The selected entries and the merged tensor.
    """
    all_checks = flat_task_checks.clone()
    updated_checks, *_ = topk_values_mask(all_checks, K=reset_thresh, return_mask=False)
    print("RESOLVING SIGN")
    final_signs = resolve_sign(updated_checks)
    assert final_signs is not None

    print(f"Disjoint AGGREGATION: {merge_func}")
    selected_entries, merged_tv = disjoint_merge_split(
        updated_checks, merge_func, final_signs
    )

    return selected_entries, merged_tv
