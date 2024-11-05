import logging
from typing import List, Mapping, Union  # noqa: F401

import numpy as np
import torch
from typing_extensions import override

from fusion_bench.method import BaseAlgorithm
from fusion_bench.modelpool import CausalLMPool
from fusion_bench.utils import timeit_context
from fusion_bench.utils.state_dict_arithmetic import state_dict_add, state_dict_mul
from fusion_bench.utils.type import StateDictType

log = logging.getLogger(__name__)


class WeightedAverageForLLama(BaseAlgorithm):
    """
    A class to perform weighted averaging of LlaMa/Mistral models.
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "normalize": "normalize",
        "weights": "weights",
        "backbone_only": "backbone_only",
        "merged_model_save_path": "merged_model_save_path",
        "save_tokenizer": "save_tokenizer",
        "push_to_hub": "push_to_hub",
    }

    def __init__(
        self,
        normalize: bool,
        weights: List[float],
        backbone_only: bool,
        merged_model_save_path: str,
        save_tokenizer: bool,
        push_to_hub: bool,
        **kwargs,
    ):
        """
        Initialize the WeightedAverageForLLama class with the given parameters.

        Args:
            normalize (bool): Whether to normalize the weights.
            weights (List[float]): The weights for averaging the models.
            backbone_only (bool): Whether to use only the backbone of the models.
            merged_model_save_path (str): The path to save the merged model.
            save_tokenizer (bool): Whether to save the tokenizer.
            push_to_hub (bool): Whether to push the model to the hub.
        """
        self.normalize = normalize
        self.weights = weights
        self.backbone_only = backbone_only
        self.merged_model_save_path = merged_model_save_path
        self.save_tokenizer = save_tokenizer
        self.push_to_hub = push_to_hub
        super().__init__(**kwargs)

    @override
    @torch.no_grad()
    def run(self, modelpool: CausalLMPool):
        """
        Executes the weighted averaging of models in the provided model pool.

        Args:
            modelpool (LLamaForCausalLMPoolThe):  pool of models to be averaged.

        Returns:
            base_model: The base model after merging the state dictionaries of the models in the pool.

        Raises:
            ValueError: If the number of weights does not match the number of models in the pool.
        """
        if modelpool.has_pretrained:
            base_model = modelpool.load_model("_pretrained_")
        else:
            base_model = modelpool.load_model(modelpool.model_names[0])

        weights = self.weights
        if len(weights) != len(modelpool.model_names):
            raise ValueError(
                "Number of weights must match the number of models.,"
                f"but got {len(weights)} weights and {len(modelpool.model_names)} models."
                f"weights: {weights}, models: {modelpool.model_names}"
            )
        if self.normalize:
            weights = np.asarray(weights)
            weights = weights / np.sum(weights)

        merged_state_dict: StateDictType = None
        for model_name, weight in zip(modelpool.model_names, weights):
            model = modelpool.load_model(model_name, backbone_only=self.backbone_only)
            sd = state_dict_mul(model.state_dict(), weight)
            if merged_state_dict is None:
                merged_state_dict = sd
            else:
                merged_state_dict = state_dict_add(merged_state_dict, sd)

        base_model.load_state_dict(
            merged_state_dict, strict=False if self.backbone_only else True
        )
        if self.merged_model_save_path is not None:
            with timeit_context(
                f"Saving the merged model to {self.merged_model_save_path}"
            ):
                modelpool.save_model(
                    base_model,
                    path=self.merged_model_save_path,
                    save_tokenizer=self.save_tokenizer,
                    push_to_hub=self.push_to_hub,
                )
        return base_model
