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

from fusion_bench import BaseAlgorithm, BaseModelPool, auto_register_config
from fusion_bench.mixins import LightningFabricMixin, SimpleProfilerMixin
from fusion_bench.models.utils import is_leaf_module, named_leaf_modules
from fusion_bench.utils import instantiate
from fusion_bench.utils.json import load_from_json, save_to_json
from fusion_bench.utils.packages import is_ray_available
from fusion_bench.utils.parameters import state_dict_to_vector
from fusion_bench.utils.state_dict_arithmetic import state_dict_sub

from .utils import frobenius_inner_product, get_task_vector_norm, svd

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


@auto_register_config
class OPCM(
    LightningFabricMixin,
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    def __init__(
        self,
        alpha: float,
        shuffle_order: bool = True,
        seed: Optional[int] = None,
        save_on_every_step: bool = True,
        evaluate_on_every_step: bool = False,
        num_ray_actors: int = 0,
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
        if self.num_ray_actors > 0:
            if is_ray_available():
                import ray
                from ray.util.actor_pool import ActorPool

                if not ray.is_initialized():
                    ray.init()

                # create actors
                if self.fabric.device.type == "cuda":
                    actor_options = {"num_gpus": 1}
                else:
                    actor_options = {}
                self.ray_actor_pool = ActorPool(
                    [
                        OPCMActor.options(**actor_options).remote(**self.config)
                        for _ in range(self.num_ray_actors)
                    ]
                )

        if self.seed is not None:
            L.seed_everything(self.seed)
        accelerator = self.fabric.device

        with self.profile("loading model"):
            pretrained_model = modelpool.load_pretrained_model()

        model_names = modelpool.model_names
        if self.shuffle_order:
            random.shuffle(model_names)

        # log the model names
        if self.log_dir is not None:
            save_to_json(model_names, Path(self.log_dir) / "model_names.json")
            tensorboard_summarywriter: "SummaryWriter" = self.tensorboard_summarywriter
            tensorboard_summarywriter.add_text(
                "global/model_names", str(model_names), global_step=0
            )

        # get the average model
        with self.profile("loading model"):
            print("Using the first model as the initial merged model.")
            merged_model = modelpool.load_model(model_names[0])
            assert merged_model is not None, "Failed to load the first model"

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

                self._layer_wise_merge(
                    merged_model=merged_model,
                    pretrained_model=pretrained_model,
                    task_model=task_model,
                    model_name=model_name,
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

        self.print_profile_summary()
        return merged_model

    def _layer_wise_merge(self, merged_model, pretrained_model, task_model, model_name):
        if self.num_ray_actors > 0:
            self._update_attributes_across_ray()

        for module_name, module in tqdm(
            list(named_leaf_modules(merged_model, ignore_empty=True)),
            desc=f"Processing {model_name}",
            leave=False,
            disable=self.num_ray_actors > 0,
        ):
            if isinstance(module, nn.Linear):
                # processing linear layers
                merge_kwargs = {
                    "merged_W": module.weight,
                    "pretrained_W": pretrained_model.get_submodule(module_name).weight,
                    "task_W": task_model.get_submodule(module_name).weight,
                    "param_name": ".".join([module_name, "weight"]),
                    "alpha": self.alpha,
                }
                if not self.num_ray_actors > 0:
                    _, merged_weight = self.merge_linear_weights(**merge_kwargs)
                    module.weight.data = merged_weight
                else:
                    if not self.ray_actor_pool.has_free():
                        returned_module_name, merged_weight = (
                            self.ray_actor_pool.get_next_unordered()
                        )
                        print(f"merged weight {returned_module_name} from ray actors.")
                        pretrained_model.get_submodule(
                            returned_module_name
                        ).weight.data = merged_weight
                    self.ray_actor_pool.submit(
                        lambda actor, kwargs: actor.merge_linear_weights.remote(
                            **kwargs
                        ),
                        merge_kwargs,
                    )
                # processing bias if exists
                if module.bias is not None:
                    module.bias.data = self.merge_other_parameters(
                        module.bias,
                        pretrained_model.get_submodule(module_name).bias,
                        task_model.get_submodule(module_name).bias,
                        param_name=".".join([module_name, "bias"]),
                    )
            else:
                for param_name, param in module.named_parameters():
                    param.data = self.merge_other_parameters(
                        merged_W=param,
                        pretrained_W=pretrained_model.get_submodule(
                            module_name
                        ).get_parameter(param_name),
                        task_W=task_model.get_submodule(module_name).get_parameter(
                            param_name
                        ),
                        param_name=".".join([module_name, param_name]),
                    )

        if self.num_ray_actors > 0:
            while self.ray_actor_pool.has_next():
                returned_module_name, merged_weight = (
                    self.ray_actor_pool.get_next_unordered()
                )
                print(f"merged weight {returned_module_name} from ray actors.")
                merged_model.get_submodule(returned_module_name).weight.data = (
                    merged_weight
                )

    def save_merged_model(self, merged_model, step: int):
        if self.log_dir is None:
            print("Log dir is None, skip saving merged model.")
            return
        os.makedirs(Path(self.log_dir) / "checkpoints", exist_ok=True)
        merged_model.save_pretrained(
            Path(self.log_dir) / "checkpoints" / f"merged_model_{step}"
        )

    def _update_attributes_across_ray(self, attr_dict=None):
        if attr_dict is None:
            # called on master
            attrs_to_sync = ["previous_lambda_t", "lambda_t"]
            assert (
                not self.ray_actor_pool.has_next()
            ), "All previous tasks must be merged before syncing attributes."

            for actor in self.ray_actor_pool._idle_actors:
                actor._update_attributes_across_ray.remote(
                    {attr: getattr(self, attr) for attr in attrs_to_sync}
                )
        else:
            # called on ray actors
            for attr, value in attr_dict.items():
                setattr(self, attr, value)

    def merge_linear_weights(
        self,
        merged_W: Tensor,
        pretrained_W: Tensor,
        task_W: Tensor,
        param_name: str,
        alpha: float,
    ):
        accelerator = self.fabric.device

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
        module_name = param_name[: param_name.rfind(".")]
        return module_name, new_merged_W.to(original_device)

    def merge_other_parameters(
        self,
        merged_W: Tensor,
        pretrained_W: Tensor,
        task_W: Tensor,
        param_name: str,
    ):
        accelerator = self.fabric.device

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


if is_ray_available():
    import ray

    OPCMActor = ray.remote(OPCM)
