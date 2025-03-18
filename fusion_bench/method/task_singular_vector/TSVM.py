"""
Example:

```bash
fusion_bench \
    method=task_singular_vector/TaskSingularVectorMerging \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20
```
"""

from typing import Iterable, List, Optional, Union

import torch
from omegaconf import ListConfig
from torch import Tensor, nn

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

    def __init__(
        self,
        alpha: Union[float, Iterable[float]] = None,
        remove_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        self.alpha = alpha
        self.remove_keys = remove_keys if remove_keys is not None else []
        super().__init__(**kwargs)

    def run(self, modelpool):
        # Load the pre-trained model and the fine-tuned models
        pretrained_model = modelpool.load_pretrained_model()
        finetuned_models = list(modelpool.models())

        ptm_check = pretrained_model.state_dict()
        ft_checks = [model.state_dict() for model in finetuned_models]
        check_parameterNamesMatch(ft_checks + [ptm_check])

        with timeit_context("Flattening out Checkpoints"):
            task_vectors = [state_dict_sub(check, ptm_check) for check in ft_checks]
            if isinstance(self.alpha, Iterable):
                assert len(self.alpha) == len(
                    task_vectors
                ), "Alpha and task vectors must have the same length"
                task_vectors = [
                    state_dict_mul(state_dict=tv, scalar=alpha)
                    for alpha, tv in zip(self.alpha, task_vectors)
                ]

        new_merged_tv = TSVM_utils.compute_and_sum_svd_mem_reduction(
            task_vectors,
            exclude_keys=self.remove_keys,
            accelerator=self.fabric.device,
        )

        # If alpha is a float, we need to scale the new merged task vector by alpha
        if self.alpha is not None and isinstance(self.alpha, float):
            print(f"Scaling new merged task vector by alpha: {self.alpha}")
            new_merged_tv = state_dict_mul(state_dict=new_merged_tv, scalar=self.alpha)

        pretrained_model.load_state_dict(
            state_dict_add(new_merged_tv, pretrained_model.state_dict())
        )
        return pretrained_model
