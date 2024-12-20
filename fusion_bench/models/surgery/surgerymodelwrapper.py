import math
from typing import TYPE_CHECKING, Callable, Generic, List, Union

import torch
from torch import nn
from transformers.models.clip.modeling_clip import (
    CLIPVisionModel,
    CLIPVisionTransformer,
)

from fusion_bench.utils.type import TorchModelType


def regularize_name(name: str):
    name = name.replace("-", "_")
    name = name.replace(".", "_")
    return name


class SurgeryModelWrapper(torch.nn.Module, Generic[TorchModelType]):

    is_surgery_model = True
    """A flag to indicate that this is a surgery model."""

    def __init__(
        self,
        model: TorchModelType,
        test_datasets: List[str],
        projection_dim: int = 512,
        hidden_dim: int = 16,
    ):
        super(SurgeryModelWrapper, self).__init__()
        self.model = model
        self.model.requires_grad_(False)

        self.test_datasets = test_datasets
        self.non_linear_func = torch.nn.ReLU()

        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim

        for dataset_name in test_datasets:
            self.add_surgery_module(dataset_name)

    def add_surgery_module(self, dataset_name: str):
        """
        Add a surgery module for a given dataset.

        Args:
            dataset_name (str): The name of the dataset.
        """
        dataset_name = regularize_name(dataset_name)

        down_proj = torch.nn.Linear(self.projection_dim, self.hidden_dim, bias=False)
        up_proj = torch.nn.Linear(self.hidden_dim, self.projection_dim, bias=False)

        torch.nn.init.kaiming_uniform_(down_proj.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(up_proj.weight)

        self.add_module(
            "feature_mapping_to_head_down_proj_{}".format(dataset_name), down_proj
        )
        self.add_module(
            "feature_mapping_to_head_up_proj_{}".format(dataset_name), up_proj
        )

    def collect_trainable_params(self):
        trainable_params = []

        # surgery parameter
        for dataset_name in self.test_datasets:
            dataset_name = regularize_name(dataset_name)
            down_proj = getattr(
                self, "feature_mapping_to_head_down_proj_{}".format(dataset_name)
            )
            up_proj = getattr(
                self, "feature_mapping_to_head_up_proj_{}".format(dataset_name)
            )
            trainable_params.append(down_proj.weight)
            trainable_params.append(up_proj.weight)
        return trainable_params

    def collect_surgery_module(self):
        surgery_module = {}

        # surgery parameter
        for dataset_name in self.test_datasets:
            dataset_name = regularize_name(dataset_name)
            down_proj = getattr(
                self, "feature_mapping_to_head_down_proj_{}".format(dataset_name)
            )
            up_proj = getattr(
                self, "feature_mapping_to_head_up_proj_{}".format(dataset_name)
            )
            surgery_module[
                "feature_mapping_to_head_down_proj_{}".format(dataset_name)
            ] = down_proj
            surgery_module[
                "feature_mapping_to_head_up_proj_{}".format(dataset_name)
            ] = up_proj

        surgery_module["non_linear_func"] = self.non_linear_func

        return surgery_module

    def compute_surgery_features(
        self,
        compute_features_fn: Union[
            torch.Tensor, Callable[[TorchModelType], torch.Tensor]
        ],
        dataset_name: str,
    ):
        """
        Compute the surgery features.

        Args:
            compute_features_fn (Union[torch.Tensor, Callable[[nn.Module], torch.Tensor]]): A function that computes the features or a tensor that represents the features.
            dataset_name (str): The name of the dataset.

        Returns:
            feature (torch.Tensor): The surgery features.
            feature0 (torch.Tensor): The original features.
            feature_sub (torch.Tensor): feature0 - feature.
        """
        dataset_name = regularize_name(dataset_name)

        if isinstance(compute_features_fn, torch.Tensor):
            feature = compute_features_fn
        elif callable(compute_features_fn):
            feature = compute_features_fn(self.model)
        else:
            raise ValueError(
                "compute_features_fn must be a tensor or a callable, but got {}".format(
                    type(compute_features_fn)
                )
            )

        feature0 = feature

        # feature bias
        down_proj = getattr(
            self, "feature_mapping_to_head_down_proj_{}".format(dataset_name)
        )
        up_proj = getattr(
            self, "feature_mapping_to_head_up_proj_{}".format(dataset_name)
        )
        feature_sub = down_proj(feature)
        feature_sub = self.non_linear_func(feature_sub)
        feature_sub = up_proj(feature_sub)

        # surgery feature
        feature = feature0 - feature_sub

        return feature, feature0, feature_sub

    def forward(self, *args, **kwargs):
        """The wrappered model should just forward like normal."""
        return self.model(*args, **kwargs)
