from typing import Dict, Literal, Optional, Union  # noqa: F401

import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import LlamaForCausalLM, LlamaModel

from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import CausalLMPool

from . import prune_utils
from .prune_utils import PruningType, find_linear_layers


def unstructured_magnitude_prune_(
    model: Union[LlamaForCausalLM, LlamaModel], sparsity_ratio: float
):
    """
    Perform unstructured magnitude pruning on the given model.

    Args:
        model (Union[LlamaForCausalLM, LlamaModel]): The model to be pruned.
        sparsity_ratio (float): The ratio of weights to be pruned.

    Returns:
        The pruned model.
    """
    if isinstance(model, LlamaForCausalLM):
        layers = model.model.layers
    elif isinstance(model, LlamaModel):
        layers = model.layers

    subset: Dict[str, nn.Linear] = find_linear_layers(layers)
    for name in tqdm(subset, desc="Pruning"):
        prune_utils.unstructured_magnitude_prune_(
            subset[name].weight,
            metric_function_or_scores=torch.rand_like,
            sparsity_ratio=sparsity_ratio,
        )

    return model


def semistructured_magnitude_prune_(
    model: Union[LlamaForCausalLM, LlamaModel], n: int, m: int
):
    """
    Perform semi-structured (N:M structured) magnitude pruning on the given model.

    Args:
        model (Union[LlamaForCausalLM, LlamaModel]): The model to be pruned.
        n (int): The number of weights to be pruned in each group.
        m (int): The total number of weights in each group.

    Returns:
        The pruned model.
    """
    if isinstance(model, LlamaForCausalLM):
        layers = model.model.layers
    elif isinstance(model, LlamaModel):
        layers = model.layers

    subset: Dict[str, nn.Linear] = find_linear_layers(layers)
    for name in tqdm(subset, desc="Pruning"):
        prune_utils.semistructured_magnitude_prune_(
            subset[name].weight,
            metric_function_or_scores=torch.rand_like,
            n=n,
            m=m,
        )

    return model


class RandomPruningForLlama(BaseAlgorithm, SimpleProfilerMixin):
    """
    A class to perform random pruning for Llama models.

    Attributes:
        prune_type (PruningType): The type of pruning to be performed.
        sparsity_ratio (float): The ratio of weights to be pruned.
        n (int): The number of weights to be pruned in each group (for semistructured pruning).
        m (int): The total number of weights in each group (for semistructured pruning).
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "prune_type": "prune_type",
        "sparsity_ratio": "sparsity_ratio",
        "n": "n",
        "m": "m",
    }

    def __init__(
        self,
        *,
        prune_type: PruningType,
        sparsity_ratio: float,
        n: int,
        m: int,
        **kwargs,
    ):
        """
        Initialize the RandomPruningForLlama class.

        Args:
            prune_type (PruningType): The type of pruning to be performed.
            sparsity_ratio (float): The ratio of weights to be pruned.
            n (int): The number of weights to be pruned in each group (for semistructured pruning).
            m (int): The total number of weights in each group (for semistructured pruning).
            **kwargs: Additional keyword arguments.
        """
        self.prune_type = prune_type
        self.sparsity_ratio = sparsity_ratio
        self.n = n
        self.m = m
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(self, modelpool: CausalLMPool):
        """
        Run the pruning algorithm on the first model from the given model pool.

        Args:
            modelpool (CausalLMPool): The pool of models to be pruned.

        Returns:
            The pruned model.
        """
        # load pre-trained model or the first model in the pool
        base_model = modelpool.load_pretrained_or_first_model()

        if self.prune_type == PruningType.UNSTRUCTURED:
            unstructured_magnitude_prune_(base_model, self.sparsity_ratio)
        elif self.prune_type == PruningType.SEMISTRUCTURED:
            semistructured_magnitude_prune_(base_model, self.n, self.m)
        else:
            raise ValueError(
                f"Invalid pruning type: {self.prune_type}"
                "Choose from 'unstructured' or 'semistructured'"
            )

        return base_model
