import math

import torch

from fusion_bench.mixins import LightningFabricMixin


class SurgeryModelWrapper(
    torch.nn.Module,
    LightningFabricMixin,
):
    def __init__(self, model, exam_datasets, Algorithm):
        super(SurgeryModelWrapper, self).__init__()
        self.model = model
        self.compute_features = Algorithm.compute_features
        self.visual_projection = Algorithm.visual_projection
        self._fabric_instance = Algorithm.fabric
        self.exam_datasets = exam_datasets
        self.non_linear_func = torch.nn.ReLU()

        self.model = self.fabric.to_device(self.model)
        for dataset_name in exam_datasets:
            # mapping
            # ViT-B/32 and ViT-B/16
            down_proj = torch.nn.Linear(512, 16, bias=False)
            up_proj = torch.nn.Linear(16, 512, bias=False)
            # ViT-L/14
            # down_proj = torch.nn.Linear(768, 16, bias=False)
            # up_proj = torch.nn.Linear(16, 768, bias=False)
            torch.nn.init.kaiming_uniform_(down_proj.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(up_proj.weight)
            down_proj = self.fabric.to_device(down_proj)
            up_proj = self.fabric.to_device(up_proj)
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
