"""
This module provides utilities for visualizing learning rate schedulers.

Functions:
    simulate_scheduler(lr_scheduler, steps): Simulates the learning rate scheduler for a given number of steps.
    plot_lr_schedulers(lr_schedulers, steps, titles): Plots the learning rates of one or more schedulers over a number of steps.
"""

from typing import TYPE_CHECKING, List, Union

import matplotlib.pyplot as plt
import torch

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler

__all__ = ["simulate_scheduler", "plot_lr_schedulers"]


def simulate_scheduler(lr_scheduler, steps: int):
    """
    Simulates the learning rate scheduler for a given number of steps.

    Args:
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler object.
        steps (int): The number of steps to simulate.

    Returns:
        List[float]: A list of learning rates for each step.
    """
    lrs = []
    for _ in range(steps):
        lr = lr_scheduler.step()
        lrs.append(lr)
    return lrs


def plot_lr_schedulers(
    lr_schedulers: Union["LRScheduler", List["LRScheduler"]],
    steps: int,
    titles: Union[str, List[str]],
    show_plot: bool = True,
):
    """
    Plots the learning rates of one or more schedulers over a number of steps.

    Args:
        lr_schedulers (Union[LRScheduler, List[LRScheduler]]): One or more learning rate scheduler objects.
        steps (int): The number of steps to simulate.
        titles (Union[str, List[str]]): Titles for the plots.

    Returns:
        fig, axes: The matplotlib figure and axes objects.
    """
    # Handle single scheduler
    if isinstance(lr_schedulers, torch.optim.lr_scheduler.LRScheduler):
        lr_schedulers = [lr_schedulers]
    if isinstance(titles, str):
        titles = [titles]

    fig, axs = plt.subplots(len(lr_schedulers), 1, figsize=(5, 3 * len(lr_schedulers)))
    if len(lr_schedulers) == 1:
        axs = [axs]

    for i, (scheduler, title) in enumerate(zip(lr_schedulers, titles)):
        lrs = simulate_scheduler(scheduler, steps)
        axs[i].plot(lrs, label=title)
        axs[i].set_title(title)
        axs[i].set_xlabel("Steps")
        axs[i].set_ylabel("Learning Rate")
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    if show_plot:
        plt.show()
    return fig, axs


# Example usage
if __name__ == "__main__":
    from fusion_bench.optim.lr_scheduler.linear_warmup import (
        CosineDecayWithWarmup,
        LinearWarmupScheduler,
        PolySchedulerWithWarmup,
    )

    # Dummy optimizer
    optimizer = torch.optim.SGD(
        [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))], lr=0.1
    )

    # Define the schedulers
    linear_scheduler = LinearWarmupScheduler(
        optimizer, t_max=100, max_lr=0.1, min_lr=0.01, init_lr=0.0, warmup_steps=10
    )
    cosine_scheduler = CosineDecayWithWarmup(
        optimizer, t_max=100, max_lr=0.1, min_lr=0.01, init_lr=0.0, warmup_steps=10
    )
    poly_scheduler = PolySchedulerWithWarmup(
        optimizer,
        t_max=100,
        max_lr=0.1,
        min_lr=0.01,
        init_lr=0.0,
        warmup_steps=40,
        poly_order=2.0,
    )

    # Plot the learning rates
    plot_lr_schedulers(
        [linear_scheduler, cosine_scheduler, poly_scheduler],
        steps=100,
        titles=[
            "Linear Warmup",
            "Cosine Decay with Warmup",
            "Poly Scheduler with Warmup",
        ],
    )
