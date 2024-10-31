import logging
import os
from typing import Dict, List, Optional, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from numpy.typing import NDArray
from torch import nn
from tqdm.auto import tqdm

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import LightningFabricMixin, SimpleProfilerMixin
from fusion_bench.utils import timeit_context
from fusion_bench.utils.parameters import (
    StateDictType,
    state_dict_to_vector,
    trainable_state_dict,
)
from fusion_bench.utils.state_dict_arithmetic import state_dict_sub

log = logging.getLogger(__name__)


class TaskVectorViolinPlot(BaseAlgorithm, LightningFabricMixin, SimpleProfilerMixin):
    R"""
    Plot violin plots of task vectors as in:
    [L.Shen, A.Tang, E.Yang et al. Efficient and Effective Weight-Ensembling Mixture of Experts for Multi-Task Model Merging](https://arxiv.org/abs/2410.21804)
    """

    # config_mapping is a mapping from the attributes to the key in the configuration files
    _config_mapping = BaseAlgorithm._config_mapping | {
        "trainable_only": "trainable_only",
        "max_points_per_model": "max_points_per_model",
        "fig_kwargs": "fig_kwargs",
        "_output_path": "output_path",
    }

    def __init__(
        self,
        trainable_only: bool,
        max_points_per_model: Optional[int] = 1000,
        fig_kwawrgs=None,
        output_path: Optional[str] = None,
        **kwargs,
    ):
        R"""
        This class creates violin plots to visualize task vectors, which represent the differences
        between fine-tuned models and their pretrained base model.

        Args:
            trainable_only (bool): If True, only consider trainable parameters when computing
                task vectors. If False, use all parameters.
            fig_kwargs (dict, optional): Dictionary of keyword arguments to pass to
                `matplotlib.pyplot.subplots`. Common options include:
                - figsize: Tuple of (width, height) in inches
                - dpi: Dots per inch
                - facecolor: Figure background color
                Defaults to None.
            output_path (str, optional): Path where the violin plot will be saved. If None,
                uses the fabric logger's log directory. Defaults to None.
            kwargs: Additional keyword arguments passed to the parent class(es).

        Example:

            ```python
            plotter = TaskVectorViolinPlot(
                trainable_only=True,
                fig_kwargs={'figsize': (10, 6), 'dpi': 300},
                output_path='./plots'
            )

            plotter.run(modelpool)
            ```
        """
        self.trainable_only = trainable_only
        self.fig_kwargs = fig_kwawrgs
        self.max_points_per_model = max_points_per_model
        self._output_path = output_path
        super().__init__(**kwargs)

    @property
    def output_path(self):
        if self._output_path is None:
            return self.fabric.logger.log_dir
        else:
            return self._output_path

    def run(self, modelpool: BaseModelPool):
        """Create violin plots of task vectors comparing different fine-tuned models against a pretrained model.

        This method implements the visualization technique from the paper "Efficient and Effective
        Weight-Ensembling Mixture of Experts for Multi-Task Model Merging". It:

        1. Loads the pretrained model
        2. Computes task vectors (differences between fine-tuned and pretrained models)
        3. Creates violin plots showing the distribution of values in these task vectors

        Args:
            modelpool (BaseModelPool): Model pool containing the pretrained model and fine-tuned models

        Returns:
            pretrained_model (nn.Model): The plot is saved to the specified output path.
        """
        assert modelpool.has_pretrained
        pretrained_model = modelpool.load_pretrained_model()

        # Compute task vectors for each fine-tuned model
        with torch.no_grad(), timeit_context("Computing task vectors"):
            task_vectors: Dict[str, NDArray] = {}
            for name, finetuned_model in tqdm(
                modelpool.named_models(), total=len(modelpool)
            ):
                print(f"computing task vectors for {name}")
                task_vectors[name] = self.get_task_vector(
                    pretrained_model, finetuned_model
                )

        # === Create violin plot ===
        fig, ax = plt.subplots(
            1, 1, **self.fig_kwargs if self.fig_kwargs is not None else {}
        )
        fig = cast(plt.Figure, fig)
        ax = cast(plt.Axes, ax)

        # Prepare data for plotting
        data = [values for values in task_vectors.values()]
        labels = list(task_vectors.keys())

        # Create violin plot using seaborn
        with timeit_context("ploting"):
            sns.violinplot(data=data, ax=ax)

        # Customize plot
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Task Vector Values")
        ax.set_title("Distribution of Task Vector Values")

        # Adjust layout to prevent label cutoff and save plot
        plt.tight_layout()
        os.makedirs(self.output_path, exist_ok=True)
        output_file = f"{self.output_path}/task_vector_violin.pdf"
        plt.savefig(output_file, bbox_inches="tight")
        plt.close(fig)

        # === Create violin plot (Abs values) ===
        fig, ax = plt.subplots(
            1, 1, **self.fig_kwargs if self.fig_kwargs is not None else {}
        )
        fig = cast(plt.Figure, fig)
        ax = cast(plt.Axes, ax)

        # Prepare data for plotting
        data = [np.abs(values) for values in task_vectors.values()]
        labels = list(task_vectors.keys())

        # Create violin plot using seaborn
        with timeit_context("ploting abs value plot"):
            sns.violinplot(data=data, ax=ax)

        # Customize plot
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("The Absolute Values")
        ax.set_title("Distribution of Task Vector Absolute Values")

        # Adjust layout to prevent label cutoff and save plot
        plt.tight_layout()
        os.makedirs(self.output_path, exist_ok=True)
        output_file = f"{self.output_path}/task_vector_violin_abs.pdf"
        plt.savefig(output_file, bbox_inches="tight")
        plt.close(fig)

        return pretrained_model

    def get_task_vector(self, pretrained_model, finetuned_model):
        task_vector = state_dict_sub(
            self.get_state_dict(finetuned_model),
            self.get_state_dict(pretrained_model),
        )
        task_vector = state_dict_to_vector(task_vector)

        task_vector = task_vector.cpu().float().numpy()
        # downsample if necessary
        if (
            self.max_points_per_model is not None
            and self.max_points_per_model > 0
            and task_vector.shape[0] > self.max_points_per_model
        ):
            log.info(
                f"Downsampling task vectors to {self.max_points_per_model} points."
            )
            indices = np.random.choice(
                task_vector.shape[0], self.max_points_per_model, replace=False
            )
            task_vector = task_vector[indices].copy()

        return task_vector

    def get_state_dict(self, model: nn.Module):
        if self.trainable_only:
            return trainable_state_dict(model)
        else:
            return model.state_dict()
