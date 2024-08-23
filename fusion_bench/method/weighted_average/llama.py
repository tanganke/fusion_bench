import logging
from collections import defaultdict
from copy import deepcopy
from typing import List, Mapping, Union

import numpy as np
import torch
from torch import Tensor, nn
from typing_extensions import override

from fusion_bench.method.base_algorithm import ModelFusionAlgorithm
from fusion_bench.modelpool import ModelPool, to_modelpool
from fusion_bench.modelpool.huggingface_llm import LLamaForCausalLMPool
from fusion_bench.utils import timeit_context
from fusion_bench.utils.state_dict_arithmetic import state_dict_add, state_dict_mul
from fusion_bench.utils.type import _StateDict


class WeightedAverageForLLama(ModelFusionAlgorithm):
    """
    A class to perform weighted averaging of models in a LLamaForCausalLMPool.

    Attributes:
        config (DictConfig): Configuration parameters for the weighted averaging process.

    Methods:
        run(modelpool: LLamaForCausalLMPool):
            Executes the weighted averaging of models in the provided model pool.
    """

    @torch.no_grad()
    @override
    def run(self, modelpool: LLamaForCausalLMPool):
        """
        Executes the weighted averaging of models in the provided model pool.

        Args:
            modelpool (LLamaForCausalLMPoolThe):  pool of models to be averaged.

        Returns:
            base_model: The base model after merging the state dictionaries of the models in the pool.

        Raises:
            ValueError: If the number of weights does not match the number of models in the pool.
        """
        config = self.config
        if modelpool.has_pretrained:
            base_model = modelpool.load_model("_pretrained_")
        else:
            base_model = modelpool.load_model(modelpool.model_names[0])

        weights = config.weights
        if len(weights) != len(modelpool.model_names):
            raise ValueError(
                "Number of weights must match the number of models.,"
                f"but got {len(weights)} weights and {len(modelpool.model_names)} models."
                f"weights: {weights}, models: {modelpool.model_names}"
            )
        if self.config.normalize:
            weights = np.asarray(weights)
            weights = weights / np.sum(weights)

        merged_state_dict = None
        for model_name, weight in zip(modelpool.model_names, weights):
            model = modelpool.load_model(model_name, backbone_only=config.backbone_only)
            sd = state_dict_mul(model.state_dict(), weight)
            if merged_state_dict is None:
                merged_state_dict = sd
            else:
                merged_state_dict = state_dict_add(merged_state_dict, sd)

        base_model.load_state_dict(
            merged_state_dict, strict=False if config.backbone_only else True
        )
        if config.merged_model_save_path is not None:
            with timeit_context(
                f"Saving the merged model to {config.merged_model_save_path}"
            ):
                modelpool.save_model(
                    base_model,
                    path=config.merged_model_save_path,
                    save_tokenizer=config.save_tokenizer,
                    push_to_hub=config.push_to_hub,
                )
        return base_model
