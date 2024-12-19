import math

import torch
from torch import nn


class SurgeryModelWrapper(torch.nn.Module):
    def __init__(
        self,
        model,
        exam_datasets,
        Algorithm,
        projection_dim: int = 512,
        hidden_dim: int = 16,
    ):
        super(SurgeryModelWrapper, self).__init__()
        self.model = model
        self.compute_features = Algorithm.compute_features
        self.visual_projection = Algorithm.visual_projection

        self.exam_datasets = exam_datasets
        self.non_linear_func = torch.nn.ReLU()

        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim

        for dataset_name in exam_datasets:
            self.add_surgery_module(dataset_name)

    def add_surgery_module(self, dataset_name: str):
        """
        Add a surgery module for a given dataset.

        Args:
            dataset_name (str): The name of the dataset.
        """
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
        for dataset_name in self.exam_datasets:
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
        for dataset_name in self.exam_datasets:
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

    def forward(self, inp, dataset_name):

        feature = self.compute_features(self.model, inp)
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

        out = None  # we do not need this
        # # classifier
        # layer_name = 'classifier_{}'.format(dataset_name)
        # classification_head = getattr(self, layer_name)
        # out = classification_head(feature)

        return out, feature, feature0, feature_sub
