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
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.parameters import (
    StateDictType,
    state_dict_to_vector,
    trainable_state_dict,
)
from fusion_bench.utils.state_dict_arithmetic import state_dict_sub

log = logging.getLogger(__name__)


class TaskVectorCosSimilarity(BaseAlgorithm, LightningFabricMixin):
    """
    This class is similar to the Dummy algorithm,
    but it also print (or save) the cosine similarity matrix between the task vectors of the models in the model pool.
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "plot_heatmap": "plot_heatmap",
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
        self.plot_heatmap = plot_heatmap
        self.trainable_only = trainable_only
        self.max_points_per_model = max_points_per_model
        self._output_path = output_path
        super().__init__(**kwargs)

    @property
    def output_path(self):
        if self._output_path is None:
            return self.fabric.logger.log_dir
        else:
            return self._output_path

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
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
        This function plots a heatmap of the provided data using seaborn.

        Args:
            data (pd.DataFrame): A pandas DataFrame containing the data to be plotted.
            figsize (tuple): A tuple specifying the size of the figure. Default is (4, 3).

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
        if self.trainable_only:
            return trainable_state_dict(model)
        else:
            return model.state_dict()
