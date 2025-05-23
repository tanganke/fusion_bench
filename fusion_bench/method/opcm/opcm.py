import os
import random
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, cast

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import CLIPVisionModel

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import LightningFabricMixin, SimpleProfilerMixin
from fusion_bench.taskpool import CLIPVisionModelTaskPool
from fusion_bench.utils import instantiate
from fusion_bench.utils.json import load_from_json, save_to_json
from fusion_bench.utils.parameters import state_dict_to_vector
from fusion_bench.utils.state_dict_arithmetic import state_dict_sub

from .utils import frobenius_inner_product, get_task_vector_norm, is_leaf_module, svd

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class OPCMForCLIP(
    BaseAlgorithm,
    LightningFabricMixin,
    SimpleProfilerMixin,
):
    def __init__(
        self,
        alpha: float,
        shuffle_order: bool = True,
        seed: Optional[int] = None,
        save_on_every_step: bool = True,
        evaluate_on_every_step: bool = False,
        **kwargs,
    ):
        """
        Continual Model Merging via SVD Projection.

        Args:
            alpha (float): the scaling factor for the SVD projection.
            shuffle_order (bool): whether to shuffle the order of the models.
            seed (Optional[int]): the seed to use.
            save_on_every_step (bool): whether to save the merged model on every step.
            evaluate_on_every_step (bool): whether to evaluate the merged model on every step.
        """
        self.alpha = alpha
        self.shuffle_order = shuffle_order
        self.seed = seed
        self.save_on_every_step = save_on_every_step
        self.evaluate_on_every_step = evaluate_on_every_step
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        if self.seed is not None:
            L.seed_everything(self.seed)
        accelerator = self.fabric.device

        with self.profile("loading model"):
            pretrained_model = modelpool.load_pretrained_model()

        model_names = modelpool.model_names
        if self.shuffle_order:
            random.shuffle(model_names)

        self.taskpool = cast(CLIPVisionModelTaskPool, self._program.taskpool)
        self._test_datasets = deepcopy(self.taskpool._test_datasets)
        """Configuration for the test datasets"""

        # log the model names
        if self.log_dir is not None:
            save_to_json(model_names, Path(self.log_dir) / "model_names.json")
            tensorboard_summarywriter: "SummaryWriter" = self.tensorboard_summarywriter
            tensorboard_summarywriter.add_text(
                "global/model_names", str(model_names), global_step=0
            )

        # get the average model
        with self.profile("loading model"):
            merged_model = modelpool.load_model(model_names[0])

        if self.evaluate_on_every_step:
            with self.profile("evaluating model"):
                self.taskpool._is_setup = False
                self.taskpool._test_datasets = DictConfig(
                    {model_names[0]: self._test_datasets[model_names[0]]}
                )
                report = self.taskpool.evaluate(deepcopy(merged_model))
                save_to_json(report, Path(self.log_dir) / "report_0.json")

        self.avg_task_vector_norm = get_task_vector_norm(merged_model, pretrained_model)
        self.all_task_vector_norm = [self.avg_task_vector_norm]
        self.fabric.log("model/task_vector_norm", self.avg_task_vector_norm, step=0)
        self.fabric.log("model/avg_task_vector_norm", self.avg_task_vector_norm, step=0)
        self.fabric.log(
            "model/merged_task_vector_norm", self.avg_task_vector_norm, step=0
        )

        self.previous_lambda_t = 1
        self.lambda_t = None
        self.fabric.log("model/lambda_t", self.previous_lambda_t, step=0)
        self.fabric.log("empirical/lambda_t", 1, step=0)

        if self.save_on_every_step:
            self.save_merged_model(merged_model, 0)

        for model_idx, model_name in tqdm(
            enumerate(model_names[1:]), desc="Processing models"
        ):
            model_idx += 1
            with self.profile("loading model"):
                task_model = modelpool.load_model(model_name)

            with self.profile("merging model"):
                self.all_task_vector_norm.append(
                    get_task_vector_norm(task_model, pretrained_model)
                )
                self.avg_task_vector_norm = np.mean(self.all_task_vector_norm)
                self.fabric.log(
                    "model/task_vector_norm",
                    self.all_task_vector_norm[-1],
                    step=model_idx,
                )
                self.fabric.log(
                    "model/avg_task_vector_norm",
                    self.avg_task_vector_norm,
                    step=model_idx,
                )

                self.lambda_t = 1  # temporary value

                for module_name, module in tqdm(
                    list(merged_model.named_modules()),
                    desc=f"Processing {model_name}",
                    leave=False,
                ):
                    if not is_leaf_module(module):
                        continue

                    if isinstance(module, nn.Linear):
                        module.weight.data = self.merge_linear_weights(
                            module.weight,
                            pretrained_model.get_submodule(module_name).weight,
                            task_model.get_submodule(module_name).weight,
                            param_name=".".join([module_name, "weight"]),
                            alpha=self.alpha,
                            accelerator=accelerator,
                        )
                        if module.bias is not None:
                            module.bias.data = self.merge_other_parameters(
                                module.bias,
                                pretrained_model.get_submodule(module_name).bias,
                                task_model.get_submodule(module_name).bias,
                                param_name=".".join([module_name, "bias"]),
                                accelerator=accelerator,
                            )
                    else:
                        for param_name, param in module.named_parameters():
                            param.data = self.merge_other_parameters(
                                merged_W=param,
                                pretrained_W=pretrained_model.get_submodule(
                                    module_name
                                ).get_parameter(param_name),
                                task_W=task_model.get_submodule(
                                    module_name
                                ).get_parameter(param_name),
                                param_name=".".join([module_name, param_name]),
                                accelerator=accelerator,
                            )

                task_vector_norm = get_task_vector_norm(merged_model, pretrained_model)
                self.lambda_t *= task_vector_norm / self.avg_task_vector_norm
                for param_name, param in merged_model.named_parameters():
                    param.data = pretrained_model.get_parameter(param_name) + (
                        param - pretrained_model.get_parameter(param_name)
                    ) * (self.avg_task_vector_norm / task_vector_norm)
                self.fabric.log("model/lambda_t", self.lambda_t, step=model_idx)
                self.fabric.log(
                    "empirical/lambda_t", np.sqrt(model_idx + 1), step=model_idx
                )
                self.previous_lambda_t = self.lambda_t
                self.lambda_t = None

                self.fabric.log(
                    "model/merged_task_vector_norm",
                    get_task_vector_norm(merged_model, pretrained_model),
                    step=model_idx,
                )

            if self.save_on_every_step:
                with self.profile("saving model"):
                    self.save_merged_model(merged_model, model_idx)

            if self.evaluate_on_every_step:
                with self.profile("evaluating model"):
                    self.taskpool._is_setup = False
                    self.taskpool._test_datasets = DictConfig(
                        {
                            n: self._test_datasets[n]
                            for n in model_names[: model_idx + 1]
                        }
                    )
                    report = self.taskpool.evaluate(deepcopy(merged_model))
                    save_to_json(
                        report, Path(self.log_dir) / f"report_{model_idx}.json"
                    )

        self.print_profile_summary()
        return merged_model

    def save_merged_model(self, merged_model: CLIPVisionModel, step: int):
        os.makedirs(Path(self.log_dir) / "checkpoints", exist_ok=True)
        merged_model.save_pretrained(
            Path(self.log_dir) / "checkpoints" / f"merged_model_{step}"
        )

    def merge_linear_weights(
        self,
        merged_W: Tensor,
        pretrained_W: Tensor,
        task_W: Tensor,
        param_name: str,
        alpha: float,
        accelerator: str = "cpu",
    ):
        original_device = merged_W.device
        merged_W = merged_W.to(accelerator)
        pretrained_W = pretrained_W.to(accelerator)
        task_W = task_W.to(accelerator)

        previous_merged_tv = merged_W - pretrained_W
        task_tv = task_W - pretrained_W

        u, s, v = svd(previous_merged_tv)
        rank = s.size(0)
        split_rank = (s.cumsum(dim=0) / s.sum() > alpha).float().argmax().item()

        projected_task_tv = u.T @ task_tv @ v
        projected_task_tv.diagonal().fill_(0)

        projected_task_tv[:split_rank, :split_rank] = 0

        cleaned_task_tv = u @ projected_task_tv @ v.T

        previous_lambda_t = self.previous_lambda_t
        lambda_t = self.lambda_t
        new_merged_W = (
            pretrained_W
            + (previous_lambda_t * previous_merged_tv + cleaned_task_tv) / lambda_t
        )
        return new_merged_W.to(original_device)

    def merge_other_parameters(
        self,
        merged_W: Tensor,
        pretrained_W: Tensor,
        task_W: Tensor,
        param_name: str,
        accelerator: str = "cpu",
    ):
        original_device = merged_W.device
        merged_W = merged_W.to(accelerator)
        pretrained_W = pretrained_W.to(accelerator)
        task_W = task_W.to(accelerator)

        previous_merged_tv = merged_W - pretrained_W
        task_tv = task_W - pretrained_W

        previous_lambda_t = self.previous_lambda_t
        lambda_t = self.lambda_t

        new_merged_W = (
            pretrained_W + (previous_lambda_t * previous_merged_tv + task_tv) / lambda_t
        )
        return new_merged_W.to(original_device)

    def compute_lambda_t(
        self, previous_merged_tv: Tensor, task_tv: Tensor, previous_lambda_t: float
    ):
        previous_merged_tv = torch.flatten(previous_merged_tv)
        task_tv = torch.flatten(task_tv)

        lambda_t = torch.linalg.vector_norm(
            previous_lambda_t * previous_merged_tv + task_tv
        ) / torch.linalg.vector_norm(previous_merged_tv)
        return lambda_t.item()
