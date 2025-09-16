from typing import List

import torch
import torch.nn as nn

from fusion_bench.utils.type import TorchModelType


def get_memory_usage(desc):
    """
    obtain the current GPU memory usage

    Returns:
        str: A string containing the allocated and cached memory in MB.
    """
    allocated = torch.cuda.memory_allocated() / 1024**2  # 转换为 MB
    cached = torch.cuda.memory_reserved() / 1024**2  # 转换为 MB
    return (
        f"{desc}\nAllocated Memory: {allocated:.2f} MB\nCached Memory: {cached:.2f} MB"
    )


@torch.no_grad()
def construct_task_wise_merged_model(
    pretrained_model: TorchModelType,
    finetuned_models: List[TorchModelType],
    clamp_weights: bool = False,
    tie_weights: bool = True,
    strict: bool = False,
):
    from fusion_bench.models.wrappers.task_wise_fusion import (
        TaskWiseMergedModel,
        get_task_wise_weights,
    )

    merging_weights = get_task_wise_weights(num_models=len(finetuned_models))
    module = TaskWiseMergedModel(
        task_wise_weight=merging_weights,
        pretrained_model=pretrained_model,
        finetuned_models=finetuned_models,
        clamp_weights=clamp_weights,
        tie_weights=tie_weights,
        strict=strict,
    )
    return module


@torch.no_grad()
def construct_layer_wise_merged_model(
    pretrained_model: TorchModelType,
    finetuned_models: List[TorchModelType],
    clamp_weights: bool = False,
    tie_weights: bool = True,
    strict: bool = False,
):
    from fusion_bench.models.wrappers.layer_wise_fusion import (
        LayerWiseMergedModel,
        get_layer_wise_weights,
    )

    merging_weights = get_layer_wise_weights(
        num_models=len(finetuned_models),
        num_layers=len([p for p in pretrained_model.parameters() if p.requires_grad]),
    )
    module = LayerWiseMergedModel(
        layer_wise_weight=merging_weights,
        pretrained_model=pretrained_model,
        finetuned_models=finetuned_models,
        clamp_weights=clamp_weights,
        tie_weights=tie_weights,
        strict=strict,
    )
    return module
