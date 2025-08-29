import logging
import os
from typing import Dict, List, Optional, cast

import numpy as np
import pandas as pd
import torch
import torch.utils
from numpy.typing import NDArray
from torch import nn
from tqdm.auto import tqdm

from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import LightningFabricMixin, auto_register_config
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.parameters import (
    StateDictType,
    state_dict_to_vector,
    trainable_state_dict,
)
from fusion_bench.utils.state_dict_arithmetic import state_dict_sub

log = logging.getLogger(__name__)


@auto_register_config
class TaskVectorCosSimilarity(
    LightningFabricMixin,
    BaseAlgorithm,
):
    """
    Computes and analyzes cosine similarity between task vectors of models in a model pool.

    This algorithm extracts task vectors from fine-tuned models by computing the difference
    between their parameters and a pretrained base model. It then calculates the pairwise
    cosine similarity between all task vectors to understand the relationships and overlap
    between different tasks.

    The task vector for a model is defined as:
        task_vector = finetuned_model_params - pretrained_model_params

    Args:
        plot_heatmap (bool): Whether to generate and save a heatmap visualization
        trainable_only (bool, optional): If True, only consider trainable parameters
            when computing task vectors. Defaults to True.
        max_points_per_model (int, optional): Maximum number of parameters to sample
            per model for memory efficiency. If None, uses all parameters.
        output_path (str, optional): Directory to save outputs. If None, uses the
            fabric logger directory.

    Outputs:
        - task_vector_cos_similarity.csv: Pairwise cosine similarity matrix
        - task_vector_cos_similarity.pdf: Heatmap visualization (if plot_heatmap=True)

    Returns:
        The pretrained model from the model pool.

    Example:
        ```python
        >>> algorithm = TaskVectorCosSimilarity(
        ...     plot_heatmap=True,
        ...     trainable_only=True,
        ...     output_path="/path/to/outputs"
        ... )
        >>> result = algorithm.run(modelpool)
        ```
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "_output_path": "output_path",
    }

    def __init__(
        self,
        plot_heatmap: bool,
        trainable_only: bool = True,
        max_points_per_model: Optional[int] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._output_path = output_path

    @property
    def output_path(self):
        if self._output_path is None:
            return self.fabric.logger.log_dir
        else:
            return self._output_path

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        """
        Execute the task vector cosine similarity analysis.

        This method:
        1. Loads the pretrained base model from the model pool
        2. Computes task vectors for each fine-tuned model
        3. Calculates pairwise cosine similarities between all task vectors
        4. Saves the similarity matrix as a CSV file
        5. Optionally generates and saves a heatmap visualization

        Args:
            modelpool (BaseModelPool): Pool containing pretrained and fine-tuned models

        Returns:
            nn.Module: The pretrained model from the model pool
        """
        pretrained_model = modelpool.load_pretrained_model()

        task_vectors = []
        for name, finetuned_model in tqdm(
            modelpool.named_models(), total=len(modelpool)
        ):
            print(f"computing task vectors for {name}")
            task_vectors.append(
                self.get_task_vector(pretrained_model, finetuned_model).to(
                    torch.float64
                )
            )
        task_vectors = torch.stack(task_vectors, dim=0)

        cos_sim_matrix = torch.zeros(
            len(modelpool), len(modelpool), dtype=torch.float64
        )
        for i in range(len(modelpool)):
            for j in range(i, len(modelpool)):
                assert task_vectors[i].size() == task_vectors[j].size()
                cos_sim_matrix[i, j] = torch.nn.functional.cosine_similarity(
                    task_vectors[i], task_vectors[j], dim=0
                )
                cos_sim_matrix[j, i] = cos_sim_matrix[i, j]

        # convert the matrix to a pandas DataFrame
        cos_sim_df = pd.DataFrame(
            cos_sim_matrix.numpy(),
            index=modelpool.model_names,
            columns=modelpool.model_names,
        )

        print(cos_sim_df)
        if self.output_path is not None:
            os.makedirs(self.output_path, exist_ok=True)
            cos_sim_df.to_csv(
                os.path.join(self.output_path, "task_vector_cos_similarity.csv")
            )

        if self.plot_heatmap:
            self._plot_heatmap(cos_sim_df)

        return pretrained_model

    def _plot_heatmap(self, data: pd.DataFrame):
        """
        Generate and save a heatmap visualization of the cosine similarity matrix.

        Creates a color-coded heatmap showing pairwise cosine similarities between
        task vectors. The heatmap is saved as a PDF file in the output directory.

        Args:
            data (pd.DataFrame): Symmetric matrix of cosine similarities between
                task vectors, with model names as both index and columns.

        Returns:
            None
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create a heatmap using seaborn
        plt.figure()
        sns.heatmap(
            data,
            annot=True,
            fmt=".2f",
            cmap="GnBu",
        )

        # Add title and labels with increased font size
        plt.title("Heatmap of Cos Similarities", fontsize=14)
        # plt.xlabel("Task", fontsize=14)
        # plt.ylabel("Task", fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)

        # Show plot
        plt.savefig(
            os.path.join(self.output_path, "task_vector_cos_similarity.pdf"),
            bbox_inches="tight",
        )
        plt.close()

    def get_task_vector(
        self, pretrained_model: nn.Module, finetuned_model: nn.Module
    ) -> torch.Tensor:
        """
        Compute the task vector for a fine-tuned model.

        The task vector represents the parameter changes from pretraining to
        fine-tuning and is computed as:
            task_vector = finetuned_params - pretrained_params

        Args:
            pretrained_model (nn.Module): The base pretrained model
            finetuned_model (nn.Module): The fine-tuned model for a specific task

        Returns:
            torch.Tensor: Flattened task vector containing parameter differences.
                If max_points_per_model is set, the vector may be downsampled.

        Note:
            - Converts parameters to float64 for numerical precision
            - Supports optional downsampling for memory efficiency
            - Uses only trainable parameters if trainable_only=True
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

        task_vector = torch.from_numpy(task_vector)
        return task_vector

    def get_state_dict(self, model: nn.Module):
        """
        Extract the state dictionary from a model.

        Args:
            model (nn.Module): The model to extract parameters from

        Returns:
            Dict[str, torch.Tensor]: State dictionary containing model parameters.
                Returns only trainable parameters if trainable_only=True,
                otherwise returns all parameters.
        """
        if self.trainable_only:
            return trainable_state_dict(model)
        else:
            return model.state_dict()
