import logging
from abc import abstractmethod
from typing import Dict, List, Literal, Optional, Tuple, cast  # noqa: F401

import torch
import torch.utils.hooks
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import LlamaForCausalLM

from fusion_bench.method import BaseAlgorithm
from fusion_bench.method.pruning.wanda_utils.data import get_loaders
from fusion_bench.method.pruning.wanda_utils.prune import prepare_calibration_input
from fusion_bench.mixins import SimpleProfilerMixin
from fusion_bench.modelpool import CausalLMPool
from fusion_bench.utils import timeit_context
from fusion_bench.utils.cache_utils import cache_to_disk

from .prune_utils import (
    PruningType,
    compute_sparsity,
    find_linear_layers,
    semistructured_magnitude_prune_,
    unstructured_magnitude_prune_,
)

log = logging.getLogger(__name__)


class BaseLoSparseHookFn:
    """
    Base class for low-sparsity hook functions.
    """

    def __init__(self, linear):
        self.linear = linear

    @abstractmethod
    def compute(self) -> Tensor:
        """
        Compute the importance scores.
        """
        pass

    @abstractmethod
    def __call__(self, linear, inp: Tuple[Tensor], out: Tensor):
        """
        Hook function to be called during the forward pass.
        """
        pass


class WandaHookFn(BaseLoSparseHookFn):
    R"""
    Here in this class, the `scalar_row` is the mean of the squared sum of the input to the linear layer along a specific input dimension.

    $$\frac{\sum_{i=1}^{N L} X_{ij}^2}{N L}$$
    """

    def __init__(self, linear: nn.Linear):
        super().__init__(linear)

        self.scalar_row = torch.zeros(
            (linear.weight.size(1),), device=linear.weight.device
        )
        self.nsamples = 0

    def compute(self):
        return torch.abs(self.linear.weight) * torch.sqrt(
            self.scalar_row.reshape(1, -1)
        )

    def __call__(self, linear: nn.Linear, inps: Tuple[Tensor], out: Tensor):
        assert len(inps) == 1
        inp = inps[0]
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        batch_size = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        # (NxL, C) -> (C, NxL)
        inp = inp.t()

        self.scalar_row *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size

        inp = inp.type(torch.float32)
        self.scalar_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples


class WandaPruningForLlama(BaseAlgorithm, SimpleProfilerMixin):
    """
    Class for Wanda pruning for Llama models.
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "nsamples": "nsamples",
        "seed": "seed",
        "use_variant": "use_variant",
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
        nsamples: int,
        seed: int,
        use_variant: bool,
        prune_type: PruningType,
        device: str,
        dtype: str,
        sparsity_ratio: float,
        n: int,
        m: int,
        model_save_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the WandaPruningForLlama class.

        Args:
            nsamples (int): Number of samples for calibration.
            seed (int): Random seed.
            use_variant (bool): Whether to use a variant of the pruning method.
            prune_type (PruningType): Type of pruning to perform.
            device (str): Device to use for computation.
            dtype (str): Data type to use for computation.
            sparsity_ratio (float): Sparsity ratio for pruning.
            n (int): Number of elements to keep in semi-structured pruning.
            m (int): Number of elements in a group for semi-structured pruning.
            model_save_path (Optional[str]): Path to save the pruned model.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.nsamples = nsamples
        self.seed = seed
        self.use_variant = use_variant
        self.prune_type = prune_type
        self.device = device
        self.dtype = dtype
        self.sparsity_ratio = sparsity_ratio
        self.n = n
        self.m = m
        self.model_save_path = model_save_path

    def run(self, modelpool: CausalLMPool):
        """
        Run the pruning algorithm on the model pool.

        Args:
            modelpool (CausalLMPool): Pool of causal language models.

        Returns:
            LlamaForCausalLM: Pruned model.
        """

        # load pre-trained model or the first model in the pool
        with self.profile("load_model"):
            model = modelpool.load_pretrained_or_first_model()
            model.seqlen = model.config.max_position_embeddings
            tokenizer = modelpool.load_tokenizer(use_fast=False)

        if not isinstance(model, (LlamaForCausalLM,)):
            log.warning(f"Model type {type(model)} may not supported.")

        inps, outs, attention_mask, position_ids = self.prepare_calibration_data(
            model, tokenizer
        )

        self.prune_using_calibration_data_(
            model,
            inps=inps,
            outs=outs,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        if self.model_save_path is not None:
            with timeit_context(f"Saving pruned model to {self.model_save_path}"):
                tokenizer.save_pretrained(self.model_save_path)
                model.save_pretrained(self.model_save_path)
        return model

    def _prepare_calibration_data(self, model, tokenizer):
        """
        Prepare calibration data for pruning.

        Args:
            model (LlamaForCausalLM): Model to be pruned.
            tokenizer: Tokenizer for the model.

        Returns:
            Tuple: Calibration data (inputs, outputs, attention mask, position IDs).
        """
        with timeit_context("loading calibration data"):
            dataloader, _ = get_loaders(
                "c4",
                nsamples=self.nsamples,
                seed=self.seed,
                seqlen=model.seqlen,
                tokenizer=tokenizer,
            )

        with torch.no_grad():
            # collect input to the first layer
            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, self.device
            )
        return inps, outs, attention_mask, position_ids

    def prepare_calibration_data(self, model: LlamaForCausalLM, tokenizer):
        """
        Prepare calibration data for pruning with caching.

        Args:
            model (LlamaForCausalLM): Model to be pruned.
            tokenizer: Tokenizer for the model.

        Returns:
            Tuple: Calibration data (inputs, outputs, attention mask, position IDs).
        """

        @cache_to_disk(
            f"outputs/cache/{model.config.name_or_path.split('/')[-1]}/calibration_data.pkl"
        )
        def _prepare_calibration_data(model, tokenizer):
            return self._prepare_calibration_data(model, tokenizer)

        return _prepare_calibration_data(model, tokenizer)

    def prune_using_calibration_data_(
        self,
        model: LlamaForCausalLM,
        *,
        inps,
        outs,
        attention_mask,
        position_ids,
    ):
        """
        Prune the model using calibration data.

        Args:
            model (LlamaForCausalLM): Model to be pruned.
            inps: Calibration inputs.
            outs: Calibration outputs.
            attention_mask: Attention mask for calibration data.
            position_ids: Position IDs for calibration data.
        """
        layers = model.model.layers
        for layer_idx, layer in tqdm(
            enumerate(layers),
            "Pruning Layers",
            total=len(layers),
            dynamic_ncols=True,
        ):
            if (
                hasattr(model, "hf_device_map")
                and f"model.layers.{layer_idx}" in model.hf_device_map
            ):
                # handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = model.hf_device_map[f"model.layers.{layer_idx}"]
                inps, outs, attention_mask, position_ids = (
                    inps.to(dev),
                    outs.to(dev),
                    attention_mask.to(dev) if attention_mask is not None else None,
                    position_ids.to(dev) if position_ids is not None else None,
                )

            # collect the importance scores
            linear_layers = cast(
                Dict[str, nn.Linear],
                find_linear_layers(layer, layers=[nn.Linear]),
            )

            # register hooks to collect the importance scores
            def get_hook_fn(linear: nn.Linear):
                hook_fn = WandaHookFn(linear)
                return hook_fn

            hooks = {}
            handles: List[torch.utils.hooks.RemovableHandle] = []
            for name, linear in linear_layers.items():
                hook_fn = get_hook_fn(linear)
                hooks[name] = hook_fn
                handles.append(linear.register_forward_hook(hook_fn))

            with torch.no_grad():
                for j in range(self.nsamples):
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]

            # compute the importance scores and remove the hooks
            metrics = {}
            for name, hook in hooks.items():
                metrics[name] = hook.compute()
            for h in handles:
                h.remove()

            # prune the weights based on the importance scores
            if self.prune_type == PruningType.UNSTRUCTURED:
                for name, linear in linear_layers.items():
                    log.info(f"Pruning {name}")
                    unstructured_magnitude_prune_(
                        linear.weight.data,
                        metrics[name],
                        sparsity_ratio=self.sparsity_ratio,
                    )
                    self.check_sparsity(linear.weight)
            elif self.prune_type == PruningType.SEMISTRUCTURED:
                for name, linear in linear_layers.items():
                    log.info(f"Pruning {name}")
                    semistructured_magnitude_prune_(
                        linear.weight.data,
                        metrics[name],
                        n=self.n,
                        m=self.m,
                    )
                    self.check_sparsity(linear.weight)
            else:
                raise ValueError(f"Invalid pruning type: {self.prune_type}")

            # compute the input to the next layer
            with torch.no_grad():
                for j in range(self.nsamples):
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
            inps, outs = outs, inps

    @torch.no_grad()
    def check_sparsity(self, weight: Tensor, tol: float = 0.01):
        """
        Check the sparsity of the weight tensor.

        Args:
            weight (Tensor): Weight tensor.
            tol (float): Tolerance for sparsity check.

        Raises:
            ValueError: If the pruning type is invalid.
        """
        if self.prune_type == PruningType.UNSTRUCTURED:
            assert (compute_sparsity(weight) - self.sparsity_ratio).abs() < tol
        elif self.prune_type == PruningType.SEMISTRUCTURED:
            assert (compute_sparsity(weight) - self.n / self.m).abs() < tol
        else:
            raise ValueError(f"Invalid pruning type: {self.prune_type}")
