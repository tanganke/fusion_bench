import logging
from typing import Callable

import numpy as np
import torch
from torch.optim.optimizer import Optimizer

log = logging.getLogger(__name__)


class MeZO(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-5,
        weight_decay: float = 0,
        eps: float = 1e-3,
    ):
        defaults = dict(eps=eps, lr=lr, weight_decay=weight_decay)
        super(MeZO, self).__init__(params, defaults)

    def step(
        self,
        closure: Callable,
        prune_ratio: float = 0.02,
    ):
        assert isinstance(closure, Callable), "closure should be provided"
        zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        self._zo_perturb_parameters(
            zo_random_seed, scaling_factor=1, prune_ratio=prune_ratio
        )

        with torch.inference_mode():
            loss1 = closure()
            if loss1 is None:
                raise ValueError("closure returned None (should return loss)")

        # Second function evaluation
        self._zo_perturb_parameters(
            zo_random_seed, scaling_factor=-2, prune_ratio=prune_ratio
        )

        with torch.inference_mode():
            loss2 = closure()

        # Compute projected gradient
        projected_grad = (loss1 - loss2) / (2 * self.defaults["eps"])
        if isinstance(projected_grad, torch.Tensor):
            projected_grad = projected_grad.item()

        # Reset model back to its parameters at start of step
        self._zo_perturb_parameters(
            zo_random_seed, scaling_factor=1, prune_ratio=prune_ratio
        )

        self._zo_update_parameters(
            zo_random_seed, projected_grad, prune_ratio=prune_ratio
        )
        return loss1, projected_grad

    def _zo_perturb_parameters(
        self,
        random_seed: int,
        scaling_factor: float,
        prune_ratio: float = 0.1,
    ):
        """
        Perturbs the parameters of the Zeroth Order Optimization algorithm.

        Args:
            random_seed (int): The random seed to use for the perturbation.
            scaling_factor (float): The scaling factor to use for the perturbation.

        Returns:
            None
        """
        torch.manual_seed(random_seed)

        for group in self.param_groups:
            for param in group["params"]:
                z = torch.normal(
                    mean=0,
                    std=1,
                    size=param.data.size(),
                    device=param.data.device,
                    dtype=param.data.dtype,
                )
                if prune_ratio < 1:
                    m = torch.bernoulli(prune_ratio * torch.ones_like(param))
                param.data = param.data + group["eps"] * scaling_factor * z * m

    def _zo_update_parameters(
        self,
        random_seed: int,
        projected_grad: float,
        prune_ratio: float = 0.1,
    ):
        # Update parameters
        torch.manual_seed(random_seed)
        for group in self.param_groups:
            for param in group["params"]:
                z = torch.normal(
                    mean=0,
                    std=1,
                    size=param.data.size(),
                    device=param.data.device,
                    dtype=param.data.dtype,
                )
                if prune_ratio < 1:
                    m = torch.bernoulli(prune_ratio * torch.ones_like(param))
                param.data = (
                    param.data
                    - group["lr"]
                    * (projected_grad * z + group["weight_decay"] * param.data)
                    * m
                )
