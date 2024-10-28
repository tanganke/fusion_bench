import pandas as pd
import torch
import torch.utils

from fusion_bench.method import BaseModelFusionAlgorithm
from fusion_bench.modelpool import BaseModelPool


class TaskVectorCosSimilarity(BaseModelFusionAlgorithm):
    """
    This class is similar to the Dummy algorithm,
    but it also print (or save) the cosine similarity matrix between the task vectors of the models in the model pool.
    """

    _config_mapping = BaseModelFusionAlgorithm._config_mapping | {
        "csv_save_path": "csv_save_path",
        "plot_heatmap": "plot_heatmap",
    }

    def __init__(self, csv_save_path: str, plot_heatmap: bool, **kwargs):
        self.csv_save_path = csv_save_path
        self.plot_heatmap = plot_heatmap
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        pretrained_model = modelpool.load_model("_pretrained_")
        pretrained_sd = torch.nn.utils.parameters_to_vector(
            pretrained_model.parameters()
        )

        task_vectors = torch.empty(len(modelpool), pretrained_sd.size(0))
        for model_idx, model_name in enumerate(modelpool.model_names):
            model = modelpool.load_model(model_name)
            model_sd = torch.nn.utils.parameters_to_vector(model.parameters())
            task_vectors[model_idx] = model_sd - pretrained_sd
        # convert the task vectors to float64
        task_vectors = task_vectors.to(dtype=torch.float64)

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
        if self.csv_save_path is not None:
            cos_sim_df.to_csv(self.csv_save_path)

        if self.plot_heatmap:
            self.plot_and_show_heatmap(self, cos_sim_df)

        return pretrained_model

    def plot_and_show_heatmap(self, data: pd.DataFrame, figsize=(4, 3)):
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
        plt.figure(figsize=figsize)
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
        plt.show()
