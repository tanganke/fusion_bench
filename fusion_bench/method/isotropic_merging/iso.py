from typing import List

import torch

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
)

from .iso_utils import check_parameterNamesMatch, iso_c, iso_cts


class IsotropicMergingInCommonSubspace(BaseAlgorithm, LightningFabricMixin):
    """
    Isotropic Merging in Common Subspace (Iso-C)
    """

    def __init__(
        self,
        scaling_factor: float,
        exclude_keys: List[str] = None,
    ):
        self.scaling_factor = scaling_factor
        self.exclude_keys = exclude_keys
        super().__init__()

    def run(self, modelpool: BaseModelPool):
        # load the pretrained model and the task vectors of all the finetuned models
        with torch.no_grad():
            pretrained_model = modelpool.load_pretrained_model()
            task_vectors = []
            for model_name in modelpool.model_names:
                finetuned_model = modelpool.load_model(model_name)
                task_vectors.append(
                    state_dict_sub(
                        finetuned_model.state_dict(), pretrained_model.state_dict()
                    )
                )
                del finetuned_model  # free memory
            check_parameterNamesMatch(task_vectors)

        # compute the merged task vector
        merged_tv = iso_c(
            task_vectors,
            accelerator=self.fabric.device,
            exclude_keys=self.exclude_keys,
        )

        # merged_parameters = pretrained_parameters + scaling_factor * merged_task_vector
        pretrained_model.load_state_dict(
            state_dict_add(
                pretrained_model.state_dict(),
                state_dict_mul(merged_tv, self.scaling_factor),
            )
        )

        return pretrained_model


class IsotropicMergingInCommonAndTaskSubspace(BaseAlgorithm, LightningFabricMixin):
    """
    Isotropic Merging in Common and Task-Specific Subspaces (Iso-CTS)
    """

    def __init__(
        self,
        scaling_factor: float,
        common_space_fraction: float,
        exclude_keys: List[str] = None,
    ):
        self.common_space_fraction = common_space_fraction
        self.scaling_factor = scaling_factor
        self.exclude_keys = exclude_keys
        super().__init__()

    def run(self, modelpool: BaseModelPool):
        # load the pretrained model and the task vectors of all the finetuned models
        with torch.no_grad():
            pretrained_model = modelpool.load_pretrained_model()
            task_vectors = []
            for model_name in modelpool.model_names:
                finetuned_model = modelpool.load_model(model_name)
                task_vectors.append(
                    state_dict_sub(
                        finetuned_model.state_dict(), pretrained_model.state_dict()
                    )
                )
                del finetuned_model  # free memory
            check_parameterNamesMatch(task_vectors)

        # compute the merged task vector
        merged_tv = iso_cts(
            task_vectors,
            common_space_fraction=self.common_space_fraction,
            accelerator=self.fabric.device,
            exclude_keys=self.exclude_keys,
        )

        # merged_parameters = pretrained_parameters + scaling_factor * merged_task_vector
        pretrained_model.load_state_dict(
            state_dict_add(
                pretrained_model.state_dict(),
                state_dict_mul(merged_tv, self.scaling_factor),
            )
        )

        return pretrained_model


ISO_C_Merge = IsotropicMergingInCommonSubspace  # alias
ISO_CTS_Merge = IsotropicMergingInCommonAndTaskSubspace  # alias
