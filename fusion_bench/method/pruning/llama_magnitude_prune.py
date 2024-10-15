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


def find_layers(module: nn.Module, layers=[nn.Linear], prefix=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        prefix (str): A prefix to add to the layer names.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    res = {}
    for name, submodule in module.named_modules(prefix=prefix):
        if isinstance(submodule, tuple(layers)):
            res[name] = submodule
    return res


def compute_sparsity(model: Union[LlamaForCausalLM, LlamaModel]):
    if isinstance(model, LlamaForCausalLM):
        layers = model.model.layers
    elif isinstance(model, LlamaModel):
        layers = model.layers

    subset: Dict[str, nn.Linear] = find_layers(layers)
    sparsity = 0
    total = 0
    for name in tqdm(subset, desc="Computing sparsity"):
        sparsity += torch.sum(subset[name].weight == 0).item()
        total += subset[name].weight.numel()

    return sparsity / total


def unstructured_magnitude_prune_(
    model: Union[LlamaForCausalLM, LlamaModel], sparsity_ratio: float, dtype, device
):
    if isinstance(model, LlamaForCausalLM):
        layers = model.model.layers
    elif isinstance(model, LlamaModel):
        layers = model.layers

    subset: Dict[str, nn.Linear] = find_layers(layers)
    for name in tqdm(subset, desc="Pruning"):
        prune_utils.unstructured_magnitude_prune_(
            subset[name].weight,
            metric_function_or_scores=torch.abs,
            sparsity_ratio=sparsity_ratio,
            dtype=dtype,
            device=device,
        )

    return model


def semistructured_magnitude_prune_(
    model: Union[LlamaForCausalLM, LlamaModel], n: int, m: int, dtype, device
):
    if isinstance(model, LlamaForCausalLM):
        layers = model.model.layers
    elif isinstance(model, LlamaModel):
        layers = model.layers

    subset: Dict[str, nn.Linear] = find_layers(layers)
    for name in tqdm(subset, desc="Pruning"):
        prune_utils.semistructured_magnitude_prune_(
            subset[name].weight,
            metric_function_or_scores=torch.abs,
            n=n,
            m=m,
            dtype=dtype,
            device=device,
        )

    return model


class MagnitudePruningForLlama(BaseModelFusionAlgorithm, SimpleProfilerMixin):
    """
    Implements magnitude-based pruning for LLama models.

    This class supports both unstructured and semistructured pruning methods.
    It loads a pre-trained model or the first model in the pool and applies the specified pruning technique.

    Methods:
        run(modelpool: LLamaForCausalLMPool) -> nn.Module:
            Executes the pruning process on the model pool and returns the pruned model.
    """

    _config_mapping = BaseModelFusionAlgorithm._config_mapping | {
        "prune_type": "prune_type",
        "device": "device",
        "dtype": "dtype",
        "sparsity_ratio": "sparsity_ratio",
        "n": "n",
        "m": "m",
    }

    def __init__(
        self,
        *,
        prune_type: Literal["unstructured", "semistructured"],
        device: str,
        dtype: Optional[str],
        sparsity_ratio: float,
        n: int,
        m: int,
        **kwargs,
    ):
        self.prune_type = prune_type
        self.device = device
        self.dtype = dtype
        self.sparsity_ratio = sparsity_ratio
        self.n = n
        self.m = m
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(self, modelpool: CausalLMPool):
        config = self.config

        # load pre-trained model or the first model in the pool
        base_model = modelpool.load_pretrained_or_first_model()

        dtype = parse_dtype(config.dtype)
        device = torch.device(config.device)

        if config.prune_type == "unstructured":
            unstructured_magnitude_prune_(
                base_model, config.sparsity_ratio, dtype=dtype, device=device
            )
        elif config.prune_type == "semistructured":
            semistructured_magnitude_prune_(
                base_model, config.n, config.m, dtype=dtype, device=device
            )
        else:
            raise ValueError(
                f"Invalid pruning type: {config.prune_type}"
                "Choose from 'unstructured' or 'semistructured'"
            )

        return base_model
