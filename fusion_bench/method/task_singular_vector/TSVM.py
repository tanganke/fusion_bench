"""
Example:

Merge 8 CLIP-ViT-B/32 models with TSVM.

```bash
fusion_bench \
    method=task_singular_vector/TaskSingularVectorMerging \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

Merge 8 CLIP-ViT-B/32 models with TSVM and return each single task model with transformed task vector.

```bash
fusion_bench \
    method=task_singular_vector/TaskSingularVectorMerging \
    method.return_single_task_models=true \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

Merge 20 CLIP-VIT-B/32 models with TSVM.

```bash
fusion_bench \
    method=task_singular_vector/TaskSingularVectorMerging \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20
```
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
    This class is used to merge multiple models with TSVM.

    see `docs/algorithms/task_singular_vector.md` for more details.
    """

    def __init__(
        self,
        alpha: Union[float, Iterable[float]] = None,
        exclude_keys: Optional[List[str]] = None,
        return_single_task_models: bool = False,
        **kwargs,
    ):
        """
        Args:
            alpha (Union[float, Iterable[float]]): The alpha value(s) to scale the transformed task vectors by.
            exclude_keys (Optional[List[str]]): The keys to exclude from the TSVM.
            return_single_task_models (bool): Whether to return the single task models after the TSVM.
        """
        self.alpha = alpha
        self.exclude_keys = exclude_keys if exclude_keys is not None else []
        self.return_single_task_models = return_single_task_models
        super().__init__(**kwargs)

    def load_pretrained_model_and_task_vectors(self, modelpool: fb.BaseModelPool):
        # Load the pre-trained model and the fine-tuned models
        pretrained_model = modelpool.load_pretrained_model()
        task_vectors = []
        for model_idx, model_name in enumerate(modelpool.model_names):
            finetuned_model = modelpool.load_model(model_name)
            task_vectors.append(
                state_dict_sub(
                    finetuned_model.state_dict(), pretrained_model.state_dict()
                )
            )
            if self.alpha is not None and isinstance(self.alpha, Iterable):
                assert len(self.alpha) == len(
                    modelpool.model_names
                ), "Alpha and task vectors must have the same length"
                task_vectors[-1] = state_dict_mul(
                    state_dict=task_vectors[-1], scalar=self.alpha[model_idx]
                )
        return pretrained_model, task_vectors

    def run(self, modelpool: fb.BaseModelPool):
        # this is the device to use for the SVD computation
        accelerator = self.fabric.device
        # Load the pre-trained model and the fine-tuned models
        pretrained_model, task_vectors = self.load_pretrained_model_and_task_vectors(
            modelpool
        )

        new_merged_tv = TSVM_utils.compute_and_sum_svd_mem_reduction(
            task_vectors,
            exclude_keys=self.exclude_keys,
            accelerator=accelerator,
            return_single_task_models=self.return_single_task_models,
        )
        if self.return_single_task_models:
            new_merged_tv, single_task_models = new_merged_tv

        # If alpha is a float, we need to scale the new merged task vector by alpha
        if self.alpha is not None and isinstance(self.alpha, (float, int)):
            print(f"Scaling new merged task vector by alpha: {self.alpha}")
            new_merged_tv = state_dict_mul(state_dict=new_merged_tv, scalar=self.alpha)

        if self.return_single_task_models:
            models = {}
            for model_idx, model_name in enumerate(modelpool.model_names):
                model = deepcopy(pretrained_model)
                model.load_state_dict(
                    state_dict_add(model.state_dict(), single_task_models[model_idx])
                )
                models[model_name] = model

        pretrained_model.load_state_dict(
            state_dict_add(new_merged_tv, pretrained_model.state_dict())
        )
        if self.return_single_task_models:
            models["merged"] = pretrained_model
            return models
        else:
            return pretrained_model
