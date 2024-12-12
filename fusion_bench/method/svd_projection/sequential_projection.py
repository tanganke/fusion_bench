import random
from collections import defaultdict
from typing import TYPE_CHECKING, List, Tuple, cast

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from tqdm.auto import tqdm

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.method import SimpleAverageAlgorithm, TaskArithmeticAlgorithm
from fusion_bench.utils import instantiate

from .utils import frobenius_inner_product, is_leaf_module, svd


class SequentialProjection(BaseAlgorithm):
    def __init__(
        self,
        scaling_factor: float,
        alpha: float,
        base_algorithm: DictConfig,
        shuffle_order: bool = True,
        accelerator: str = "auto",
        **kwargs,
    ):
        self.scaling_factor = scaling_factor
        self.alpha = alpha
        self.shuffle_order = shuffle_order
        self._base_algorithm = base_algorithm
        self.accelerator = accelerator
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        if self.accelerator == "auto":
            accelerator = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            accelerator = self.accelerator

        pretrained_model = modelpool.load_pretrained_model()

        model_names = modelpool.model_names
        if self.shuffle_order:
            random.shuffle(model_names)

        self.previous_lambda_t = defaultdict(lambda: 1)
        self.lambda_t = {}

        # get the average model
        merged_model = modelpool.load_model(model_names[0])

        for model_idx, model_name in tqdm(
            enumerate(model_names[1:]), desc="Processing models"
        ):
            task_model = modelpool.load_model(model_name)
            for module_name, module in tqdm(
                list(merged_model.named_modules()),
                desc=f"Processing {model_name}",
                leave=False,
            ):
                if not is_leaf_module(module):
                    continue

                if isinstance(module, nn.Linear):
                    (
                        module.weight.data,
                        self.lambda_t[".".join([module_name, "weight"])],
                    ) = self.merge_linear_weights(
                        module.weight,
                        pretrained_model.get_submodule(module_name).weight,
                        task_model.get_submodule(module_name).weight,
                        previous_lambda_t=self.previous_lambda_t[
                            ".".join([module_name, "weight"])
                        ],
                        alpha=self.alpha,
                        accelerator=accelerator,
                    )
                    if module.bias is not None:
                        (
                            module.bias.data,
                            self.lambda_t[".".join([module_name, "bias"])],
                        ) = self.merge_other_parameters(
                            module.bias,
                            pretrained_model.get_submodule(module_name).bias,
                            task_model.get_submodule(module_name).bias,
                            previous_lambda_t=self.previous_lambda_t[
                                ".".join([module_name, "bias"])
                            ],
                            accelerator=accelerator,
                        )
                else:
                    for param_name, param in module.named_parameters():
                        (
                            param.data,
                            self.lambda_t[".".join([module_name, param_name])],
                        ) = self.merge_other_parameters(
                            merged_W=param,
                            pretrained_W=pretrained_model.get_submodule(
                                module_name
                            ).get_parameter(param_name),
                            task_W=task_model.get_submodule(module_name).get_parameter(
                                param_name
                            ),
                            previous_lambda_t=self.previous_lambda_t[
                                ".".join([module_name, param_name])
                            ],
                            accelerator=accelerator,
                        )

            self.previous_lambda_t = self.lambda_t
            self.lambda_t = {}

        return merged_model

    def merge_linear_weights(
        self,
        merged_W: Tensor,
        pretrained_W: Tensor,
        task_W: Tensor,
        previous_lambda_t: float,
        alpha: float,
        accelerator: str = "cpu",
    ):
        original_device = merged_W.device
        merged_W = merged_W.to(accelerator)
        pretrained_W = pretrained_W.to(accelerator)
        task_W = task_W.to(accelerator)

        merged_tv = merged_W - pretrained_W
        task_tv = task_W - pretrained_W

        lambda_t = self.compute_lambda_t(merged_tv, task_tv, previous_lambda_t)

        u, s, v = svd(merged_tv)
        rank = s.size(0)
        split_rank = (s.cumsum(dim=0) / s.sum() > alpha).float().argmax().item()

        interference_u = u[:, :split_rank]
        interference_v = v[:, :split_rank]

        cleaned_task_tv = (
            task_tv
            - interference_u
            @ interference_u.T
            @ task_tv
            @ interference_v
            @ interference_v.T
        )

        new_merged_W = (
            pretrained_W + (previous_lambda_t * merged_tv + cleaned_task_tv) / lambda_t
        )
        return new_merged_W.to(original_device), lambda_t

    def merge_other_parameters(
        self,
        merged_W: Tensor,
        pretrained_W: Tensor,
        task_W: Tensor,
        previous_lambda_t: float,
        accelerator: str = "cpu",
    ):
        original_device = merged_W.device
        merged_W = merged_W.to(accelerator)
        pretrained_W = pretrained_W.to(accelerator)
        task_W = task_W.to(accelerator)

        previous_merged_tv = merged_W - pretrained_W
        task_tv = task_W - pretrained_W

        lambda_t = self.compute_lambda_t(previous_merged_tv, task_tv, previous_lambda_t)

        new_merged_W = (
            pretrained_W + (previous_lambda_t * previous_merged_tv + task_tv) / lambda_t
        )
        return new_merged_W.to(original_device), lambda_t

    def compute_lambda_t(
        self, previous_merged_tv: Tensor, task_tv: Tensor, previous_lambda_t: float
    ):
        previous_merged_tv = torch.flatten(previous_merged_tv)
        task_tv = torch.flatten(task_tv)

        lambda_t = torch.linalg.vector_norm(
            previous_lambda_t * previous_merged_tv + task_tv
        ) / torch.linalg.vector_norm(previous_merged_tv)
        return lambda_t
