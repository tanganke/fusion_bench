from typing import Literal, Optional, Union

import torch
from torch import Dict, Tensor, nn
from tqdm.auto import tqdm
from transformers import LlamaForCausalLM, LlamaModel

from fusion_bench.method import BaseModelFusionAlgorithm
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import CausalLMPool
from fusion_bench.utils.dtype import parse_dtype

from . import prune_utils
from .prune_utils import PruningType, find_linear_layers


def unstructured_magnitude_prune_(
    model: Union[LlamaForCausalLM, LlamaModel], sparsity_ratio: float
):
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


class RandomPruningForLlama(BaseModelFusionAlgorithm, SimpleProfilerMixin):
    _config_mapping = BaseModelFusionAlgorithm._config_mapping | {
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
        self.prune_type = prune_type
        self.sparsity_ratio = sparsity_ratio
        self.n = n
        self.m = m
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(self, modelpool: CausalLMPool):
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
