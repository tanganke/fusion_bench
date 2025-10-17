"""
Continual Model Merging without Data: Dual Projections for Balancing Stability and Plasticity. NeurIPS, 2025.


Example:

fusion_bench \
    method=dop/dop \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
"""

import logging
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, cast

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.autograd import Variable
from tqdm.auto import tqdm
from transformers import CLIPVisionModel

from fusion_bench import BaseAlgorithm, BaseModelPool, auto_register_config
from fusion_bench.method.simple_average import simple_average
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.taskpool import CLIPVisionModelTaskPool
from fusion_bench.utils import seed_everything_by_time
from fusion_bench.utils.json import save_to_json

from .min_norm_solvers import MinNormSolver, gradient_normalizers
from .utils import is_leaf_module, svd

log = logging.getLogger(__name__)


@auto_register_config
class ContinualDOPForCLIP(BaseAlgorithm, LightningFabricMixin):

    def __init__(
        self,
        seed: Optional[int] = None,
        shuffle_order: bool = False,
        save_on_every_step: bool = True,
        evaluate_on_every_step: bool = False,
        lr: float = 1e-4,
        num_steps: int = 200,
        mgda: bool = True,
        ema: bool = True,
        ema_beta: float = 0.99,
        alpha: float = None,
        svd_epsilon: float = 1.0,
        svd_proj_space: str = "uv",
        **kwargs,
    ):
        self.lr = lr
        self.num_steps = num_steps
        self.mgda = mgda
        self.ema = ema
        self.ema_beta = ema_beta
        self.alpha = alpha
        self.svd_epsilon = svd_epsilon
        self.svd_proj_space = svd_proj_space
        self.seed = seed
        self.shuffle_order = shuffle_order
        self.save_on_every_step = save_on_every_step
        self.evaluate_on_every_step = evaluate_on_every_step

        assert (
            self.svd_epsilon >= 0 and self.svd_epsilon <= 1
        ), "The svd_epsilon should be in the range of [0, 1]"
        assert (
            self.alpha >= 0 and self.alpha <= 1
        ), "The alpha should be in the range of [0, 1]"
        super().__init__(**kwargs)

    def print_params(self, pretrained_model):
        total_params = 0
        linear_params = 0
        linear_weight_params = 0
        for module_name, module in pretrained_model.named_modules():
            if not is_leaf_module(module):
                continue
            if isinstance(module, nn.Linear):
                linear_params += sum(p.numel() for n, p in module.named_parameters())
                linear_weight_params += sum(
                    p.numel() for n, p in module.named_parameters() if "weight" in n
                )
            total_params += sum(p.numel() for p in module.parameters())

        linear_ratio = linear_params / total_params * 100
        linear_weight_ratio = linear_weight_params / total_params * 100
        print(f"Total Parameters: {total_params}")
        print(f"Linear Parameters: {linear_params}")
        print(f"Linear Weight Parameters: {linear_weight_params}")
        print(f"Linear Ratio: {linear_ratio:.2f}%")
        print(f"Linear Weight Ratio: {linear_weight_ratio:.2f}%")

    def run(self, modelpool: BaseModelPool):
        if self.seed is not None:
            L.seed_everything(self.seed)
        else:
            seed_everything_by_time(self.fabric)

        # get the model names, shuffle if needed
        # the model names will be saved to the log directory as `model_names.json`
        model_names = modelpool.model_names
        if self.shuffle_order:
            random.shuffle(model_names)
        if self.log_dir is not None:
            save_to_json(model_names, os.path.join(self.log_dir, "model_names.json"))

        if self.evaluate_on_every_step:
            """Configuration for the test datasets"""
            self.taskpool = cast(CLIPVisionModelTaskPool, self._program.taskpool)
            self._test_datasets = deepcopy(self.taskpool._test_datasets)

        pretrained_model = modelpool.load_pretrained_model()

        merged_model = None
        for model_idx, model_name in enumerate(model_names):
            print(
                f"--------- Optimizing {model_idx + 1}/{len(model_names)}-th with {model_name} ---------"
            )
            if model_idx == 0:
                merged_model = modelpool.load_model(model_names[0])
            else:
                merged_model = self._layer_wise_optimize(
                    model_names=["merged", model_name],
                    pretrained_model=deepcopy(pretrained_model),
                    finetuned_models={
                        "merged": merged_model,
                        model_name: modelpool.load_model(model_name),
                    },
                    model_idx=model_idx,
                )

            if self.save_on_every_step:
                self.save_merged_model(merged_model, model_idx)

            if self.evaluate_on_every_step:
                self.taskpool._is_setup = False
                self.taskpool._test_datasets = DictConfig(
                    {n: self._test_datasets[n] for n in model_names[: model_idx + 1]}
                )
                report = self.taskpool.evaluate(deepcopy(merged_model))
                save_to_json(report, Path(self.log_dir) / f"report_{model_idx}.json")

        return merged_model

    def _layer_wise_optimize(
        self,
        model_names: List[str],
        pretrained_model: nn.Module,
        finetuned_models: Dict[str, nn.Module],
        model_idx: int,
    ):
        time_cost = []
        for module_name, module in pretrained_model.named_modules():
            if not is_leaf_module(module):
                continue

            if isinstance(module, nn.Linear):
                if module.weight.requires_grad:
                    import time

                    start_time = time.time()
                    merged_weight = self._optimize_weight(
                        module.weight,
                        {
                            model_name: finetuned_models[model_name]
                            .get_submodule(module_name)
                            .weight
                            for model_name in model_names
                        },
                        module_name,
                        model_idx,
                    )
                    end_time = time.time()
                    time_cost.append(end_time - start_time)
                    module.weight.data = merged_weight.data
                else:
                    module.weight.data = simple_average(
                        [
                            finetuned_models[model_name]
                            .get_submodule(module_name)
                            .weight
                            for model_name in model_names
                        ]
                    )
                if module.bias is not None:
                    module.bias.data = simple_average(
                        [
                            finetuned_models[model_name].get_submodule(module_name).bias
                            for model_name in model_names
                        ]
                    )
            else:
                simple_average(
                    [
                        finetuned_models[model_name].get_submodule(module_name)
                        for model_name in model_names
                    ],
                    base_module=module,
                )

        return pretrained_model

    def _optimize_weight(
        self,
        pretrained_weight: Tensor,
        finetuned_weights: Dict[str, Tensor],
        module_name: str,
        model_idx: int,
    ):
        assert (
            self.fabric.world_size == 1
        ), "This algorithm is not currently supported in distributed training"

        pretrained_weight = self.fabric.to_device(pretrained_weight.detach())
        finetuned_weights = {
            model_name: self.fabric.to_device(finetuned_weight.detach())
            for model_name, finetuned_weight in finetuned_weights.items()
        }

        merged_weight = self.fabric.to_device(
            nn.Parameter(
                simple_average(
                    [
                        finetuned_weight.detach()
                        for finetuned_weight in finetuned_weights.values()
                    ]
                ),
                requires_grad=True,
            )
        )

        # Compute SVD of the difference between the finetuned and pretrained weights
        proj_u_dict = {}
        proj_v_dict = {}
        proj_s_dict = {}
        for i, finetuned_weight in enumerate(finetuned_weights.values()):
            finetuned_tv = finetuned_weight - pretrained_weight
            u, s, v = svd(finetuned_tv, full_matrices=True)
            epsilon = 1.0 if self.svd_epsilon > 1.0 else self.svd_epsilon
            cumsum_ratio = s.cumsum(dim=0) / s.sum()
            split_rank = torch.searchsorted(cumsum_ratio, epsilon).item()
            u_main = u[:, :split_rank]
            v_main = v[:, :split_rank]
            s_main = s[:split_rank]
            proj_u_dict[i] = u_main
            proj_v_dict[i] = v_main
            proj_s_dict[i] = s_main

        if self.mgda:
            if self.ema:
                ema_sol = [self.alpha, 1 - self.alpha]
            # This is multiple-gradient descent algorithm (MGDA) optimization
            optimizer = torch.optim.Adam([merged_weight], lr=self.lr)
            all_losses = [[], []]
            all_alphas = [[], []]
            for step_idx in tqdm(
                range(self.num_steps), desc=f"Optimizing {module_name} weight"
            ):
                # Scaling the loss functions based on the algorithm choice
                loss_data = {}
                grads = {}
                for i, finetuned_weight in enumerate(finetuned_weights.values()):
                    proj_u = proj_u_dict[i]
                    proj_v = proj_v_dict[i]
                    proj_s = proj_s_dict[i]
                    delta_tv = merged_weight - finetuned_weight
                    loss_i = self.cal_loss_i(delta_tv, proj_s, proj_u, proj_v)
                    loss_data[i] = float(loss_i.data)

                    all_losses[i].append(float(loss_i.data))

                    optimizer.zero_grad()
                    loss_i.backward()
                    grads[i] = Variable(
                        merged_weight.grad.data.clone(), requires_grad=False
                    )

                # Normalize all gradients
                gn = gradient_normalizers(
                    grads=grads, losses=loss_data, normalization_type="loss"
                )
                for i, _ in enumerate(finetuned_weights.values()):
                    grads[i] = grads[i] / float(gn[i])

                # Frank-Wolfe iteration to compute scales.
                sol, min_norm = MinNormSolver.find_min_norm_element(
                    [[grads[i]] for i in range(len(finetuned_weights.values()))]
                )

                if self.ema:
                    ema_sol = [
                        self.ema_beta * ema_sol[i] + (1 - self.ema_beta) * float(sol[i])
                        for i in range(len(sol))
                    ]
                    sol = ema_sol
                    all_alphas[0].append(ema_sol[0])
                    all_alphas[1].append(ema_sol[1])

                # Scaled back-propagation
                loss = 0
                for i, finetuned_weight in enumerate(finetuned_weights.values()):
                    # Comptue gradients of each loss function wrt parameters
                    proj_u = proj_u_dict[i]
                    proj_v = proj_v_dict[i]
                    proj_s = proj_s_dict[i]
                    delta_tv = merged_weight - finetuned_weight
                    loss_i = self.cal_loss_i(delta_tv, proj_s, proj_u, proj_v)
                    loss += float(sol[i]) * loss_i

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        else:
            # This is a naive weighted optimization
            optimizer = torch.optim.Adam([merged_weight], lr=self.lr)
            for step_idx in tqdm(
                range(self.num_steps), desc=f"Optimizing {module_name} weight"
            ):
                loss = 0
                for i, finetuned_weight in enumerate(finetuned_weights.values()):
                    proj_u = proj_u_dict[i]
                    proj_v = proj_v_dict[i]
                    proj_s = proj_s_dict[i]
                    delta_tv = merged_weight - finetuned_weight
                    loss_i = self.cal_loss_i(delta_tv, proj_s, proj_u, proj_v)
                    loss += self.alpha * loss_i if i == 0 else (1 - self.alpha) * loss_i

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return merged_weight.detach().cpu()

    def cal_loss_i(self, delta_tv, proj_s, proj_u, proj_v):
        proj_delta_1 = torch.diag(proj_s) @ proj_u.T @ delta_tv
        proj_delta_2 = delta_tv @ proj_v @ torch.diag(proj_s)
        loss_i_u = torch.linalg.matrix_norm(proj_delta_1, ord="fro") ** 2
        loss_i_v = torch.linalg.matrix_norm(proj_delta_2, ord="fro") ** 2
        if self.svd_proj_space == "uv":
            loss_i = loss_i_u + loss_i_v
        elif self.svd_proj_space == "u":
            loss_i = loss_i_u
        elif self.svd_proj_space == "v":
            loss_i = loss_i_v
        else:
            raise ValueError("Invalid svd_proj_space")

        return loss_i

    def save_merged_model(self, merged_model: CLIPVisionModel, step: int):
        os.makedirs(Path(self.log_dir) / "checkpoints", exist_ok=True)
        merged_model.save_pretrained(
            Path(self.log_dir) / "checkpoints" / f"merged_model_{step}"
        )
