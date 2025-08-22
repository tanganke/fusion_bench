import copy
import logging
import math
import os
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from transformers import PreTrainedModel

import fusion_bench
from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import SimpleProfilerMixin
from fusion_bench.models import create_default_model_card
from fusion_bench.utils.type import StateDictType

log = logging.getLogger(__name__)

EPS = 1e-8


def compute_angle(
    state_dict_1: StateDictType,
    state_dict_2: StateDictType,
    ref_state_dict: StateDictType,
    ignore_keys: List[str] = [],
    return_cos: bool = False,
) -> Dict[str, float]:
    """
    Compute the angle between two state dictionaries relative to a reference state dictionary.

    Args:
        state_dict_1: First state dictionary
        state_dict_2: Second state dictionary
        ref_state_dict: Reference state dictionary (typically pre-trained model)
        ignore_keys: Keys to ignore during computation
        return_cos: If True, return cosine values instead of angles in degrees

    Returns:
        Dictionary mapping parameter names to angles (in degrees) or cosine values
    """
    # Remove the keys not used for CLIP fine-tuning (from the notebook example)

    return_dict = OrderedDict()

    with torch.no_grad():
        for key in ref_state_dict:
            if key in ignore_keys:
                log.info(f"Ignoring key '{key}'")
                continue

            state_dict_1_val = state_dict_1[key]
            state_dict_2_val = state_dict_2[key]
            ref_val = ref_state_dict[key]

            if not (state_dict_1_val.shape == state_dict_2_val.shape == ref_val.shape):
                log.warning(
                    f"Shape mismatch for key '{key}', ignored during merging: "
                    f"({state_dict_1_val.shape}, {state_dict_2_val.shape}, {ref_val.shape})"
                )
                continue

            vector1 = (state_dict_1_val - ref_val).clone().detach()
            vector2 = (state_dict_2_val - ref_val).clone().detach()

            vector1 = vector1.float()
            vector2 = vector2.float()

            cosine_val = torch.sum(vector1 * vector2) / (
                math.sqrt(torch.sum(vector1**2) * torch.sum(vector2**2)) + EPS
            )
            cosine_val = torch.clamp(
                cosine_val, min=-1.0, max=1.0
            )  # Prevent nan from acos

            if return_cos:
                return_dict[key] = cosine_val.item()
            else:
                return_dict[key] = np.rad2deg(
                    torch.acos(cosine_val).detach().cpu().item()
                )

    return return_dict


def compute_ratio(angle_dict: Dict[str, float], k: int = 2) -> Dict[str, float]:
    """
    Compute interpolation ratios based on angles between fine-tuned models.

    Args:
        angle_dict: Dictionary mapping parameter names to angles in degrees
        k: Number of fine-tuned models (default: 2)

    Returns:
        Dictionary mapping parameter names to interpolation ratios
    """
    ratio_dict = {}
    for key in angle_dict.keys():
        angle = np.deg2rad(angle_dict[key])
        ratio_dict[key] = k * np.cos(angle) / ((k - 1) * np.cos(angle) + 1 + EPS)
    return ratio_dict


def merge_weights(
    w1: StateDictType, w2: StateDictType, w0: StateDictType, ratio: Dict[str, float]
) -> StateDictType:
    """
    Merge model weights using ModelStock formula.

    Args:
        w1: First fine-tuned model weights
        w2: Second fine-tuned model weights
        w0: Pre-trained model weights
        ratio: Interpolation ratios for each parameter

    Returns:
        Merged model weights
    """
    # Compute w12 = (w1 + w2) / 2
    w12 = {}
    for key in w1.keys():
        w12[key] = (w1[key].clone() + w2[key].clone()) / 2.0

    # Apply ModelStock formula: w_merge = t * w12 + (1-t) * w0
    w_merge = copy.deepcopy(w12)
    for key, r in ratio.items():
        w_merge[key] = w12[key].clone() * r + w0[key].clone() * (1.0 - r)

    return w_merge


@fusion_bench.auto_register_config
class ModelStock(SimpleProfilerMixin, BaseAlgorithm):
    """
    Model Stock: All we need is just a few fine-tuned models

    This method merges fine-tuned models by interpolating between their average
    and a pre-trained anchor model, with interpolation ratios determined by
    the angle between fine-tuned models in parameter space.
    """

    def __init__(
        self,
        ignore_keys: Optional[List[str]] = None,
        model_save_path: Optional[str] = None,
        model_save_kwargs: Optional[DictConfig] = None,
        **kwargs,
    ):
        """
        Initialize ModelStock algorithm.

        Args:
            ignore_keys: Additional parameter keys to ignore during merging
        """
        super().__init__(**kwargs)
        if self.ignore_keys is None:
            self.ignore_keys = []
        if self.model_save_kwargs is None:
            self.model_save_kwargs = DictConfig({})

    def run(self, modelpool: BaseModelPool) -> nn.Module:
        """
        Run the ModelStock merging algorithm.

        Args:
            modelpool: Pool of models containing pre-trained and fine-tuned models

        Returns:
            Merged model
        """
        with self.profile("model loading"):
            # Load the pre-trained model (anchor)
            pretrained_model = modelpool.load_pretrained_model()
            if isinstance(pretrained_model, fusion_bench.LazyStateDict):
                assert (
                    pretrained_model.meta_module is not None
                ), "Meta module is not initialized"
            pretrained_state_dict = pretrained_model.state_dict()

            # Load all fine-tuned models
            finetuned_models = []
            finetuned_state_dicts = []

            for model_name in modelpool.model_names:
                model = modelpool.load_model(model_name)
                finetuned_models.append(model)
                finetuned_state_dicts.append(model.state_dict())
                log.info(f"Loaded fine-tuned model: {model_name}")

        if len(finetuned_models) < 2:
            raise ValueError("ModelStock requires at least 2 fine-tuned models")

        log.info(f"Running ModelStock with {len(finetuned_models)} fine-tuned models")

        with self.profile("compute angles and ratios"):
            if len(finetuned_models) == 2:
                # Two fine-tuned models case
                angle_dict = compute_angle(
                    finetuned_state_dicts[0],
                    finetuned_state_dicts[1],
                    pretrained_state_dict,
                    ignore_keys=self.ignore_keys,
                )
                ratio_dict = compute_ratio(angle_dict, k=2)

                log.info(f"Computed angles for {len(angle_dict)} parameter groups")

            else:
                # N fine-tuned models case - compute average angle
                angles_sum = {}
                angles_count = {}

                # Compute pairwise angles and average them
                for i in range(len(finetuned_models)):
                    for j in range(i + 1, len(finetuned_models)):
                        angle_dict = compute_angle(
                            finetuned_state_dicts[i],
                            finetuned_state_dicts[j],
                            pretrained_state_dict,
                            ignore_keys=self.ignore_keys,
                        )

                        for key, angle in angle_dict.items():
                            if key not in angles_sum:
                                angles_sum[key] = 0
                                angles_count[key] = 0
                            angles_sum[key] += angle
                            angles_count[key] += 1

                # Average the angles
                avg_angle_dict = {}
                for key in angles_sum:
                    avg_angle_dict[key] = angles_sum[key] / angles_count[key]

                ratio_dict = compute_ratio(avg_angle_dict, k=len(finetuned_models))

                log.info(
                    f"Computed average angles for {len(avg_angle_dict)} parameter groups"
                )

        with self.profile("merge weights"):
            if len(finetuned_models) == 2:
                # Direct merging for two models
                merged_state_dict = merge_weights(
                    finetuned_state_dicts[0],
                    finetuned_state_dicts[1],
                    pretrained_state_dict,
                    ratio_dict,
                )
            else:
                # For N models, first compute the average of fine-tuned models
                avg_finetuned_state_dict = {}
                for key in finetuned_state_dicts[0].keys():
                    avg_finetuned_state_dict[key] = torch.zeros_like(
                        finetuned_state_dicts[0][key]
                    )
                    for state_dict in finetuned_state_dicts:
                        avg_finetuned_state_dict[key] += state_dict[key]
                    avg_finetuned_state_dict[key] /= len(finetuned_state_dicts)

                # Apply ModelStock formula: w_H = t * w_avg + (1-t) * w_0
                merged_state_dict = copy.deepcopy(avg_finetuned_state_dict)
                for key, r in ratio_dict.items():
                    merged_state_dict[key] = avg_finetuned_state_dict[
                        key
                    ].clone() * r + pretrained_state_dict[key].clone() * (1.0 - r)

        # Load merged weights into the model
        if isinstance(pretrained_model, nn.Module):
            result_model = pretrained_model
        elif isinstance(pretrained_model, fusion_bench.LazyStateDict):
            result_model = deepcopy(pretrained_model.meta_module)
            result_model.to(device=pretrained_model._device)
        result = result_model.load_state_dict(merged_state_dict, strict=False)

        if result.unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys in state dict: {result.unexpected_keys}"
            )
        if result.missing_keys:
            log.warning(f"Missing keys in state dict: {result.missing_keys}")

        if self.model_save_path is not None:
            with self.profile("model saving"):
                modelpool.save_model(
                    model, path=self.model_save_path, **self.model_save_kwargs
                )
                if isinstance(model, PreTrainedModel):
                    modelcard = create_default_model_card(
                        models=[
                            modelpool.get_model_path(m)
                            for m in modelpool.all_model_names
                        ],
                        description="Merged model using [Model Stock](https://arxiv.org/abs/2403.19522).",
                        algorithm_config=self.config,
                        modelpool_config=modelpool.config,
                    )
                    with open(
                        os.path.join(self.model_save_path, "README.md"), "w"
                    ) as f:
                        f.write(modelcard)

        self.print_profile_summary()
        log.info("ModelStock merging completed successfully")
        return result_model
