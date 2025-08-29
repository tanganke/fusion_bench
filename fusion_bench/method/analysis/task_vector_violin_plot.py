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

from fusion_bench import BaseAlgorithm, BaseModelPool, StateDictType, timeit_context
from fusion_bench.mixins import (
    LightningFabricMixin,
    SimpleProfilerMixin,
    auto_register_config,
)
from fusion_bench.utils import state_dict_to_vector, trainable_state_dict
from fusion_bench.utils.state_dict_arithmetic import state_dict_sub

log = logging.getLogger(__name__)


@auto_register_config
class TaskVectorViolinPlot(
    LightningFabricMixin,
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    """
    Creates violin plots to visualize the distribution of task vector values across models.

    This class implements the task vector visualization technique described in:
    "Efficient and Effective Weight-Ensembling Mixture of Experts for Multi-Task Model Merging"
    by L. Shen, A. Tang, E. Yang et al. (https://arxiv.org/abs/2410.21804)

    Task vectors represent the parameter differences between fine-tuned models and their
    pretrained base model, computed as:
        task_vector = finetuned_params - pretrained_params

    The algorithm generates two types of violin plots:
    1. Distribution of raw task vector values (positive and negative)
    2. Distribution of absolute task vector values

    Args:
        trainable_only (bool): If True, only consider trainable parameters when computing
            task vectors. If False, use all parameters.
        max_points_per_model (int, optional): Maximum number of parameters to sample
            per model for memory efficiency. If None or 0, uses all parameters.
            Defaults to 1000.
        fig_kwargs (dict, optional): Dictionary of keyword arguments to pass to
            matplotlib.pyplot.subplots. Common options include:
            - figsize: Tuple of (width, height) in inches
            - dpi: Dots per inch for resolution
            - facecolor: Figure background color
            Defaults to None.
        output_path (str, optional): Directory to save the violin plots. If None,
            uses the fabric logger's log directory. Defaults to None.

    Outputs:
        - task_vector_violin.pdf: Violin plot of raw task vector value distributions
        - task_vector_violin_abs.pdf: Violin plot of absolute task vector value distributions

    Returns:
        The pretrained model from the model pool.

    Example:
        ```python
        plotter = TaskVectorViolinPlot(
            trainable_only=True,
            max_points_per_model=5000,
            fig_kwargs={'figsize': (12, 8), 'dpi': 300},
            output_path='./analysis_plots'
        )
        pretrained_model = plotter.run(modelpool)
        ```

    Note:
        This visualization is particularly useful for understanding:
        - How different tasks affect model parameters
        - The magnitude and distribution of parameter changes
        - Similarities and differences between task adaptations
    """

    # config_mapping is a mapping from the attributes to the key in the configuration files
    _config_mapping = BaseAlgorithm._config_mapping | {
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
        """
        Initialize the TaskVectorViolinPlot analyzer.

        Args:
            trainable_only (bool): Whether to consider only trainable parameters when
                computing task vectors. Set to True to focus on learnable parameters,
                False to include all parameters including frozen ones.
            max_points_per_model (int, optional): Maximum number of parameter values
                to sample per model for visualization. Useful for large models to
                manage memory usage and plot clarity. Set to None or 0 to use all
                parameters. Defaults to 1000.
            fig_kwargs (dict, optional): Keyword arguments passed to matplotlib's
                subplots function for plot customization. Examples:
                - {'figsize': (10, 6)} for plot dimensions
                - {'dpi': 300} for high resolution
                - {'facecolor': 'white'} for background color
                Defaults to None (uses matplotlib defaults).
            output_path (str, optional): Directory path where violin plots will be saved.
                If None, uses the fabric logger's log directory. The directory will be
                created if it doesn't exist. Defaults to None.
            **kwargs: Additional keyword arguments passed to parent classes.

        Note:
            The parameter name 'fig_kwawrgs' appears to be a typo for 'fig_kwargs'.
            This should be corrected in the parameter name for consistency.
        """
        super().__init__(**kwargs)
        self._output_path = output_path

    @property
    def output_path(self):
        if self._output_path is None:
            return self.fabric.logger.log_dir
        else:
            return self._output_path

    def run(self, modelpool: BaseModelPool):
        """
        Execute the task vector violin plot analysis and visualization.

        This method implements the core algorithm that:
        1. Loads the pretrained base model from the model pool
        2. Computes task vectors for each fine-tuned model (parameter differences)
        3. Creates two violin plots showing the distribution of task vector values:
           - Raw values plot: Shows positive and negative parameter changes
           - Absolute values plot: Shows magnitude of parameter changes
        4. Saves both plots as PDF files in the output directory

        The visualization technique follows the approach described in:
        "Efficient and Effective Weight-Ensembling Mixture of Experts for Multi-Task Model Merging"

        Args:
            modelpool (BaseModelPool): Pool containing both a pretrained model and
                fine-tuned models. Must have `has_pretrained=True`.

        Returns:
            nn.Module: The pretrained model loaded from the model pool.

        Raises:
            AssertionError: If the model pool doesn't contain a pretrained model.

        Side Effects:
            - Creates output directory if it doesn't exist
            - Saves 'task_vector_violin.pdf' (raw values distribution)
            - Saves 'task_vector_violin_abs.pdf' (absolute values distribution)
            - Prints progress information during task vector computation

        Example Output Files:
            - task_vector_violin.pdf: Shows how parameters change (+ and -)
            - task_vector_violin_abs.pdf: Shows magnitude of parameter changes
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
        """
        Compute the task vector representing parameter changes from pretraining to fine-tuning.

        The task vector quantifies how model parameters have changed during task-specific
        fine-tuning and is computed as:
            task_vector = finetuned_params - pretrained_params

        Args:
            pretrained_model (nn.Module): The base pretrained model
            finetuned_model (nn.Module): The fine-tuned model for a specific task

        Returns:
            np.ndarray: Flattened numpy array containing parameter differences.
                If max_points_per_model is set, the array may be randomly downsampled
                for memory efficiency and visualization clarity.

        Processing Steps:
            1. Extract state dictionaries from both models
            2. Compute parameter differences (subtraction)
            3. Flatten to 1D vector
            4. Convert to numpy array with float32 precision
            5. Optionally downsample if max_points_per_model is specified

        Note:
            - Uses only trainable parameters if trainable_only=True
            - Downsampling uses random sampling without replacement
            - Preserves the relative distribution of parameter changes
        """
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
        """
        Extract the state dictionary from a model based on parameter filtering settings.

        Args:
            model (nn.Module): The PyTorch model to extract parameters from

        Returns:
            Dict[str, torch.Tensor]: State dictionary containing model parameters.
                If trainable_only=True, returns only parameters with requires_grad=True.
                If trainable_only=False, returns all parameters including frozen ones.

        Note:
            This method respects the trainable_only configuration to focus analysis
            on either learnable parameters or the complete parameter set depending
            on the research question being addressed.
        """
        if self.trainable_only:
            return trainable_state_dict(model)
        else:
            return model.state_dict()
