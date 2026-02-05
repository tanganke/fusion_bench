"""
Continual Model Merging without Data: Dual Projections for Balancing Stability and Plasticity. NeurIPS, 2025.
(Architecture agnostic implementation)
"""

import logging
import os
import random
import time
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

from fusion_bench import BaseAlgorithm, BaseModelPool, auto_register_config
from fusion_bench.method.simple_average import simple_average
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.models.utils import named_leaf_modules
from fusion_bench.utils import seed_everything_by_time
from fusion_bench.utils.dtype import dtype_support_svd
from fusion_bench.utils.json import save_to_json
from fusion_bench.utils.packages import is_ray_available

from .min_norm_solvers import MinNormSolver, gradient_normalizers
from .utils import is_leaf_module, print_params, svd

log = logging.getLogger(__name__)


@auto_register_config
class DOPMerging(BaseAlgorithm, LightningFabricMixin):
    """
    Dual Projections for Balancing Stability and Plasticity (DOP) merging algorithm.

    This method implements continual model merging without data by using dual projections
    in the SVD space to balance stability (preserving previously merged model's knowledge)
    and plasticity (incorporating new model's knowledge).

    The algorithm merges models sequentially, optimizing each merge using gradient descent
    with optional multi-gradient descent algorithm (MGDA) for better trade-offs.

    Reference:
        Continual Model Merging without Data: Dual Projections for Balancing Stability and Plasticity.
        NeurIPS, 2025.
    """

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
        exclude_keys: List[str] | None = None,
        num_ray_actors: int = 0,
        **kwargs,
    ):
        """
        Initialize the DOP merging algorithm.

        Args:
            seed: Random seed for reproducibility. If None, uses time-based seeding.
            shuffle_order: Whether to shuffle the order of models before merging.
            save_on_every_step: Whether to save the model after each merge step.
            evaluate_on_every_step: Whether to evaluate the model after each merge step.
            lr: Learning rate for the optimization process.
            num_steps: Number of optimization steps per layer.
            mgda: Whether to use Multi-Gradient Descent Algorithm for balancing losses.
            ema: Whether to use exponential moving average for MGDA weights.
            ema_beta: EMA decay rate for MGDA weights (only used if ema=True).
            alpha: Weight for balancing between stability and plasticity (0-1).
                   When mgda=False, used as a fixed weight. When mgda=True with ema=True,
                   used as initial weight.
            svd_epsilon: Threshold for SVD rank selection (0-1). Determines how much
                        variance to preserve in the projection space.
            svd_proj_space: SVD projection space to use: 'u', 'v', or 'uv' (both).
            exclude_keys: List of module names to exclude from optimization.
            num_ray_actors: Number of Ray actors to use for parallel processing. If 0, ray is not used.
            **kwargs: Additional arguments passed to BaseAlgorithm.
        """
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

        if exclude_keys is None:
            exclude_keys = []
        self.exclude_keys = exclude_keys

        assert (
            self.svd_epsilon >= 0 and self.svd_epsilon <= 1
        ), "The svd_epsilon should be in the range of [0, 1]"
        assert (
            self.alpha >= 0 and self.alpha <= 1
        ), "The alpha should be in the range of [0, 1]"
        super().__init__(**kwargs)

    def run(self, modelpool: BaseModelPool):
        """
        Execute the DOP merging algorithm on a pool of models.

        Merges models sequentially, where each new model is merged with the
        previously merged result. The first model is used as-is, and subsequent
        models are merged using layer-wise optimization.

        Args:
            modelpool: The model pool containing models to merge and the pretrained model.

        Returns:
            The final merged model after sequentially merging all models in the pool.
        """
        if self.num_ray_actors > 0:
            if is_ray_available():
                import ray
                from ray.util.actor_pool import ActorPool

                if not ray.is_initialized():
                    ray.init()

                # create actors
                self.ray_actor_pool = ActorPool(
                    [
                        DOPMergingActor.remote(**self.config)
                        for _ in range(self.num_ray_actors)
                    ]
                )
            else:
                raise ImportError(
                    "Ray is not installed. Please install ray to use this feature."
                )

        model_names = modelpool.model_names
        if self.shuffle_order:
            random.shuffle(model_names)

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

        return merged_model

    def _optimize_linear_layer(
        self,
        module_name: str,
        module: nn.Linear,
        finetuned_weights: Dict[str, nn.Linear],
        model_idx: int,
    ):
        if module.weight.requires_grad and module_name not in self.exclude_keys:
            original_dtype = module.weight.dtype
            merged_weight = self._optimize_weight(
                module.weight,
                finetuned_weights,
                module_name,
                model_idx,
            )
            merged_weight = merged_weight.to(dtype=original_dtype)
        else:
            merged_weight = simple_average(list(finetuned_weights.values()))
        return module_name, merged_weight

    def _layer_wise_optimize(
        self,
        model_names: List[str],
        pretrained_model: nn.Module,
        finetuned_models: Dict[str, nn.Module],
        model_idx: int,
    ):
        """
        Optimize model parameters layer by layer.

        Iterates through all leaf modules in the pretrained model and merges their weights
        with the corresponding modules in the finetuned models. Linear layers with trainable
        weights (not in exclude_keys) are optimized using gradient descent, while other
        parameters are simply averaged.

        Args:
            model_names: List of model names to merge (e.g., ['merged', 'new_model']).
            pretrained_model: The base pretrained model (structure modified in-place).
            finetuned_models: Dictionary mapping model names to their finetuned versions.
            model_idx: Index of the current model being merged (for tracking/logging).

        Returns:
            The pretrained_model with optimized/merged weights from finetuned models.
        """
        for module_name, module in named_leaf_modules(pretrained_model):
            finetuned_modules = {
                model_name: finetuned_models[model_name].get_submodule(module_name)
                for model_name in model_names
            }
            if isinstance(module, nn.Linear):
                # process weight
                finetuned_weights = {
                    model_name: finetuned_modules[model_name].weight
                    for model_name in model_names
                }
                if self.num_ray_actors == 0:
                    _, merged_weight = self._optimize_linear_layer(
                        module_name,
                        module=module,
                        finetuned_weights=finetuned_weights,
                        model_idx=model_idx,
                    )
                    module.weight.data = merged_weight.data
                else:
                    if not self.ray_actor_pool.has_free():
                        module_name, merged_weight = (
                            self.ray_actor_pool.get_next_unordered()
                        )
                        pretrained_model.get_submodule(module_name).weight.data = (
                            merged_weight
                        )
                    self.ray_actor_pool.submit(
                        lambda actor, kwargs: actor._optimize_linear_layer.remote(
                            *kwargs
                        ),
                        {
                            "module_name": module_name,
                            "module": module,
                            "finetuned_weights": finetuned_weights,
                            "model_idx": model_idx,
                        },
                    )

                # process bias if exists
                if module.bias is not None:
                    module.bias.data = simple_average(
                        [m.bias for m in finetuned_modules.values()]
                    )
            else:
                simple_average(list(finetuned_modules.values()), base_module=module)

        if self.num_ray_actors > 0:
            while self.ray_actor_pool.has_next():
                module_name, merged_weight = self.ray_actor_pool.get_next_unordered()
                pretrained_model.get_submodule(module_name).weight.data = merged_weight

        return pretrained_model

    def _optimize_weight(
        self,
        pretrained_weight: Tensor,
        finetuned_weights: Dict[str, Tensor],
        module_name: str,
        model_idx: int,
    ):
        """
        Optimize a single weight matrix by balancing projections in SVD space.

        Performs gradient-based optimization to find merged weights that minimize
        the projection loss in the SVD space of task vectors. Uses either MGDA
        for automatic weight balancing or fixed alpha weighting.

        The algorithm:
        1. Computes SVD of each task vector (finetuned - pretrained)
        2. Projects the difference between merged and finetuned weights onto SVD subspaces
        3. Optimizes merged weights to minimize projection losses

        Args:
            pretrained_weight: The original pretrained weight matrix.
            finetuned_weights: Dictionary mapping model names to their finetuned weight matrices.
            module_name: Name of the module being optimized (for logging).
            model_idx: Index of the current model being merged (for tracking).

        Returns:
            Optimized merged weight matrix on CPU.
        """
        assert (
            self.fabric.world_size == 1
        ), "This algorithm is not currently supported in distributed training"

        with torch.no_grad():
            # Convert weights to float if original dtype does not support SVD
            original_dtype = pretrained_weight.dtype
            if not dtype_support_svd(original_dtype):
                pretrained_weight = pretrained_weight.float()
                finetuned_weights = {
                    model_name: finetuned_weight.float()
                    for model_name, finetuned_weight in finetuned_weights.items()
                }

            # Move weights to the appropriate device
            pretrained_weight = self.fabric.to_device(pretrained_weight.detach())
            finetuned_weights = {
                model_name: self.fabric.to_device(finetuned_weight.detach())
                for model_name, finetuned_weight in finetuned_weights.items()
            }

            # Initialize merged weight as simple average of finetuned weights
            merged_weight = self.fabric.to_device(
                nn.Parameter(
                    simple_average(list(finetuned_weights.values())), requires_grad=True
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

        return merged_weight.detach().to(dtype=original_dtype, device="cpu")

    def cal_loss_i(self, delta_tv, proj_s, proj_u, proj_v):
        """
        Calculate the projection loss for a single task.

        Computes the Frobenius norm of the projection of the weight difference
        onto the SVD subspace(s) defined by U and/or V matrices.

        Args:
            delta_tv: Difference between merged weight and finetuned weight (task vector difference).
            proj_s: Singular values from SVD of the task vector.
            proj_u: Left singular vectors (U) from SVD.
            proj_v: Right singular vectors (V) from SVD.

        Returns:
            Scalar loss value representing the projection distance.
        """
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


if is_ray_available():
    import ray

    DOPMergingActor = ray.remote(DOPMerging)
