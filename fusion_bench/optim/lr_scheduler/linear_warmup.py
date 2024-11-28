"""
Modified from pytorch_optimizer: https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/lr_scheduler/linear_warmup.py
"""

import math
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch

from fusion_bench.optim.exception import NegativeLRError, NegativeStepError

__all__ = [
    "BaseLinearWarmupScheduler",
    "LinearWarmupScheduler",
    "CosineDecayWithWarmup",
    "PolySchedulerWithWarmup",
]


class BaseLinearWarmupScheduler(ABC):
    r"""BaseLinearWarmupScheduler class.

    The LR Scheduler class based on this class has linear warmup strategy.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer. It will set learning rate to all trainable parameters in optimizer.
        T_max (int): Total steps to train.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate.
        init_lr (float): Initial learning rate.
        warmup_steps (int): Steps to warm-up.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        max_lr: float,
        min_lr: float = 0.0,
        init_lr: float = 0.0,
        warmup_steps: int = 0,
    ):
        """
        Initialize the BaseLinearWarmupScheduler.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer to apply the learning rate schedule.
            T_max (int): Total number of training steps.
            max_lr (float): Maximum learning rate.
            min_lr (float): Minimum learning rate.
            init_lr (float): Initial learning rate.
            warmup_steps (int): Number of steps for the warm-up phase.
        """
        self.optimizer = optimizer
        self.total_steps = T_max
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps

        self.step_t: int = 0
        self.base_lrs: List[float] = []

        # record current value in self._last_lr to match API from torch.optim.lr_scheduler
        self.last_lr: List[float] = [init_lr]

        self.validate_parameters()

        self._init_lr()

    def validate_parameters(self):
        """
        Validate the parameters to ensure they are non-negative.

        Raises:
            NegativeLRError: If any of the learning rates are negative.
            NegativeStepError: If any of the step values are negative.
        """
        if self.min_lr < 0:
            raise NegativeLRError(self.min_lr, "min_lr")

        if self.max_lr < 0:
            raise NegativeLRError(self.max_lr, "max_lr")

        if self.init_lr < 0:
            raise NegativeLRError(self.init_lr, "init_lr")

        if self.total_steps < 0:
            raise NegativeStepError(self.total_steps, "T_max")

        if self.warmup_steps < 0:
            raise NegativeStepError(self.warmup_steps, "warmup_steps")

    def _init_lr(self):
        """
        Initialize the learning rate for each parameter group in the optimizer.
        """
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def step(self):
        """
        Update the learning rate for the current step.

        Returns:
            float: The updated learning rate.
        """
        if self.step_t < self.warmup_steps:
            value = (
                self.init_lr
                + (self.max_lr - self.init_lr) * self.step_t / self.warmup_steps
            )
        elif self.step_t == self.warmup_steps:
            value = self.max_lr
        else:
            value = self._step()

        self.step_t += 1

        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = value

        self.last_lr = [value]

        return value

    @abstractmethod
    def _step(self) -> float:  # pragma: no cover
        """
        Abstract method to calculate the learning rate for the current step.

        Returns:
            float: The calculated learning rate.
        """
        raise NotImplementedError

    def get_lr(self) -> float:
        """
        Get the current learning rate.

        Returns:
            float: The current learning rate.
        """
        return self.last_lr[0]


class LinearWarmupScheduler(BaseLinearWarmupScheduler):
    r"""Linear LR Scheduler w/ linear warmup."""

    def _step(self) -> float:
        """
        Calculate the learning rate for the current step using a linear decay.

        Returns:
            float: The calculated learning rate.
        """
        return self.max_lr + (self.min_lr - self.max_lr) * (
            self.step_t - self.warmup_steps
        ) / (self.total_steps - self.warmup_steps)


class CosineDecayWithWarmup(BaseLinearWarmupScheduler):
    r"""Cosine LR Scheduler w/ linear warmup."""

    def _step(self) -> float:
        """
        Calculate the learning rate for the current step using a cosine decay.

        Returns:
            float: The calculated learning rate.
        """
        phase: float = (
            (self.step_t - self.warmup_steps)
            / (self.total_steps - self.warmup_steps)
            * math.pi
        )
        return self.min_lr + (self.max_lr - self.min_lr) * (np.cos(phase) + 1.0) / 2.0


class PolySchedulerWithWarmup(BaseLinearWarmupScheduler):
    r"""Poly LR Scheduler.

    Args:
        poly_order (float): LR scheduler decreases with steps.
    """

    def __init__(self, optimizer, poly_order: float = 0.5, **kwargs):
        """
        Initialize the PolySchedulerWithWarmup.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer to apply the learning rate schedule.
            poly_order (float): Order of the polynomial for the learning rate decay.
            kwargs: Additional arguments for the base class.

        Raises:
            ValueError: If poly_order is not positive.
        """
        self.poly_order = poly_order

        if poly_order <= 0:
            raise ValueError(f"[-] poly_order must be positive. {poly_order}")

        super().__init__(optimizer, **kwargs)

    def _step(self) -> float:
        """
        Calculate the learning rate for the current step using a polynomial decay.

        Returns:
            float: The calculated learning rate.
        """
        return (
            self.min_lr
            + (self.max_lr - self.min_lr)
            * (self.step_t - self.warmup_steps) ** self.poly_order
        )
