import logging
from copy import deepcopy
from typing import List, Mapping, Union  # noqa: F401

import torch
from torch import nn
from tqdm.autonotebook import tqdm

from fusion_bench.method import BaseAlgorithm
from fusion_bench.modelpool import BaseModelPool

log = logging.getLogger(__name__)


class DepthUpscalingAlgorithm(BaseAlgorithm):
    R"""
    Implements the Depth Upscaling Algorithm.

    - Kim et al. SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling. http://arxiv.org/abs/2312.15166

    This class extends the `BaseModelFusionAlgorithm` to handle depth upscaling of models.
    It supports upscaling the depth of a model by duplicating specified layers.

    Args:
        layer_indices (list): List of layer indices to duplicate.
        **kwargs: Additional keyword arguments.
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "layer_indices": "layer_indices",
    }

    def __init__(self, layer_indices: Union[str, List[int]], **kwargs):
        self.layer_indices = layer_indices
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(self, modelpool: nn.ModuleList | BaseModelPool) -> nn.ModuleList:
        """
        Executes the depth upscaling algorithm on a given model pool.

        This method checks the type of the model pool, ensures that it contains only one model, and verifies that the model is an instance of `nn.ModuleList`.

        Args:
            modelpool (nn.ModuleList | ModelPool): The pool of models to upscale. Must contain only one model.

        Returns:
            nn.ModuleList: The upscaled model.

        Raises:
            AssertionError: If the model pool contains more than one model or if the model is not an instance of `nn.ModuleList`.
            ValueError: If an invalid layer specification is provided in the configuration.
        """
        # check the modelpool type
        if isinstance(modelpool, BaseModelPool):
            assert len(modelpool) == 1, "DepthUpscaling only support one model"
            model = modelpool.load_model(modelpool.model_names[0])
            assert isinstance(
                model, nn.ModuleList
            ), f"The model should be a `nn.ModuleList`, but got {type(model)}"
        elif isinstance(modelpool, nn.ModuleList):
            model = modelpool
        else:
            raise AssertionError(
                f"Invalid modelpool type: {type(modelpool)}. Expected `ModelPool` or `nn.ModuleList`."
            )

        # parse the layers
        layer_indices = self.layer_indices
        parsed_layer_indices = []
        for layer in layer_indices:
            if isinstance(layer, int):
                parsed_layer_indices.append(layer)
            elif isinstance(layer, str):
                parsed_layer_indices.extend(eval(layer))
            else:
                raise ValueError("Invalid layer specification: {}".format(layer))

        # create a new model with the specified layers
        new_model = nn.ModuleList(
            [
                deepcopy(model[i])
                for i in tqdm(
                    parsed_layer_indices, desc="constructing depth-upscaled model"
                )
            ]
        )

        return new_model
