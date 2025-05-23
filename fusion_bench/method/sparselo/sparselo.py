import gc
import logging
from abc import abstractmethod
from copy import deepcopy
from typing import Dict, List, Literal, Optional, Tuple, Union, cast

import lightning as L
import numpy as np
import torch
import torch.utils.hooks
from accelerate import init_empty_weights
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import LlamaForCausalLM
from typing_extensions import override

from fusion_bench.method import BaseAlgorithm
from fusion_bench.method.pruning.llama_wanda_prune import WandaHookFn
from fusion_bench.method.pruning.prune_utils import (
    PruningType,
    compute_sparsity,
    find_linear_layers,
    semistructured_magnitude_prune_,
    unstructured_magnitude_prune_,
)
from fusion_bench.method.pruning.wanda_utils.data import get_loaders
from fusion_bench.method.pruning.wanda_utils.prune import prepare_calibration_input
from fusion_bench.mixins import SimpleProfilerMixin
from fusion_bench.modelpool import CausalLMPool
from fusion_bench.models.modeling_losparse_llama import LoSparseLlamaForCausalLM
from fusion_bench.models.modeling_losparse_llama.losparse_linear import LoSparseLinear
from fusion_bench.models.modeling_losparse_llama.utils import convert_to_losparse_llama
from fusion_bench.utils import cache_to_disk, print_parameters, timeit_context
from fusion_bench.utils.devices import get_device
from fusion_bench.utils.dtype import get_dtype

log = logging.getLogger(__name__)


@torch.no_grad()
def extract_low_rank_part_(linear: LoSparseLinear, rank: int):
    assert isinstance(
        linear, LoSparseLinear
    ), f"Expected LoSparseLinear, got {type(linear)}"

    u, s, vh = cast(
        Tuple[Tensor, Tensor, Tensor],
        torch.linalg.svd(linear.weight.float(), full_matrices=False),
    )
    v = vh.T
    uk = u[:, :rank]
    sk = s[:rank]
    vk = v[:, :rank]
    linear.lo_A.data = vk.T.to(linear.lo_A.dtype).contiguous()
    linear.lo_B.data = (uk * sk).to(linear.lo_B.dtype).contiguous()
    linear.weight.data = (linear.weight - linear.lo_B @ linear.lo_A).contiguous()
    return linear


def iterative_weight_update(w, w_pruned, mask, rank):
    w_diff = w - w_pruned
    u, s, vh = torch.linalg.svd(w_diff.float(), full_matrices=False)
    v = vh.t()
    rank = min(s.size(0) - 1, rank)
    uk = u[:, rank:]
    sk = s[rank:]
    vk = v[:, rank:]
    w_pruned = w_pruned + (mask * (uk @ torch.diag(sk) @ vk.t())).to(w_pruned.dtype)
    spectrum_ratio = torch.sum(s[:rank]) / torch.sum(s)
    return (w_pruned, spectrum_ratio)


def pcp_loss_with_mask(w, q, mask):
    _lambda = 1 / np.sqrt(np.max(w.size()))
    nuclear_loss = torch.linalg.matrix_norm((w * (~mask) + q * mask).float(), ord="nuc")
    l1_loss = _lambda * torch.linalg.matrix_norm((w * mask - q * mask).float(), ord=1)
    return nuclear_loss + l1_loss


def PCP_search_with_mask(w, mask, T_max=1000, lr=1e-2):
    q = torch.zeros_like(w).float().requires_grad_(True)
    optimizer = torch.optim.AdamW([q], lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=1e-1 * lr
    )
    for step_idx in tqdm(range(T_max)):
        optimizer.zero_grad()
        loss = pcp_loss_with_mask(w, q, mask)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if step_idx % (T_max // 20) == 0:
            print(f"Step {step_idx}: Loss {loss.item()}")
    s = (w * mask - q * mask).to(w.dtype)
    return s


class SparseLoForLlama(BaseAlgorithm, SimpleProfilerMixin):
    "Zero-Shot SVD Algorithm"

    _variants_requires_calibration_data = ["wanda"]
    _variants_hook_mapping = {"wanda": WandaHookFn}

    _config_mapping = BaseAlgorithm._config_mapping | {
        "nsamples": "nsamples",
        "seed": "seed",
        "rank": "rank",
        "sparsity_ratio": "sparsity_ratio",
        "prune_type": "prune_type",
        "n": "n",
        "m": "m",
        "device": "device",
        "variant": "variant",
    }

    def __init__(
        self,
        *,
        nsamples: int,
        variant: Literal["dense", "random", "wanda", "lowrank-only", "magnitude"],
        seed: int,
        rank: int,
        sparsity_ratio: float,
        prune_type: PruningType,
        n: int,
        m: int,
        device: Optional[str] = None,
        model_save_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nsamples = nsamples
        self.variant = variant
        self.seed = seed
        self.rank = rank
        self.sparsity_ratio = sparsity_ratio
        self.prune_type = prune_type
        self.device = device
        self.model_save_path = model_save_path
        self.n = n
        self.m = m

    @override
    def run(self, modelpool: CausalLMPool):
        self.modelpool = modelpool
        if self.seed is not None:
            L.seed_everything(self.seed)

        # load pre-trained model or the first model in the pool
        with self.profile("load_model"):
            model = modelpool.load_pretrained_or_first_model()
            model.seqlen = model.config.max_position_embeddings
            tokenizer = modelpool.load_tokenizer(use_fast=False)

        if not isinstance(model, (LlamaForCausalLM,)):
            log.warning(f"Model type {type(model)} may not supported.")

        if self.variant in self._variants_requires_calibration_data:
            inps, outs, attention_mask, position_ids = self.prepare_calibration_data(
                model, tokenizer
            )

        model = convert_to_losparse_llama(model, rank=self.rank)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for linear in find_linear_layers(model, layers=[LoSparseLinear]).values():
            linear = cast(LoSparseLinear, linear)
            linear.lo_A.data.zero_()
            linear.lo_B.data.zero_()
            linear.skip_lowrank = True

        match self.variant:
            case "dense":
                # this variant is a no-op, just for debug the conversion
                pass
            case "lowrank-only":
                self.extract_low_rank_parts_(model)
                self.set_weights_to_zeros_(model)
            case "random":
                self.random_prune_(model)
            case "magnitude":
                self.magnitude_prune_(model)
            case variant if variant in self._variants_requires_calibration_data:
                self.prune_using_calibration_data_(
                    model,
                    inps=inps,
                    outs=outs,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
            case _:
                raise ValueError(f"Invalid variant: {self.variant}")

        if self.model_save_path is not None:
            with timeit_context(f"Saving the model to {self.model_save_path}"):
                tokenizer.save_pretrained(self.model_save_path)
                model.save_pretrained(self.model_save_path)

        return model

    def set_weights_to_zeros_(self, model):
        layers: nn.ModuleList = model.model.layers
        for layer in tqdm(
            list(layers),
            "Pruning Layers",
            dynamic_ncols=True,
        ):
            for name, losparse_linear in layer.named_modules():
                if isinstance(losparse_linear, LoSparseLinear):
                    log.info(f"Pruning {name}, set weights to zeros")
                    losparse_linear.weight.data.zero_()

    @torch.no_grad()
    def extract_low_rank_parts_(self, model):
        for layer in tqdm(
            list(model.model.layers),
            "Extract Low-Rank Parts (Layers)",
            dynamic_ncols=True,
        ):
            for losparse_linear in layer.modules():
                if isinstance(losparse_linear, LoSparseLinear):
                    if self.device is not None:
                        original_device = get_device(losparse_linear)
                        losparse_linear.to(self.device)
                    extract_low_rank_part_(losparse_linear, self.rank)
                    if self.device is not None:
                        losparse_linear.to(original_device)

    def _prepare_calibration_data(self, model, tokenizer):
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

        @cache_to_disk(
            f"outputs/cache/{model.config.name_or_path.split('/')[-1]}/calibration_data.pkl"
        )
        def _prepare_calibration_data(model, tokenizer):
            return self._prepare_calibration_data(model, tokenizer)

        return _prepare_calibration_data(model, tokenizer)

    def random_prune_(self, model):
        layers: nn.ModuleList = model.model.layers
        for layer in tqdm(
            list(layers),
            "Pruning Layers",
            dynamic_ncols=True,
        ):
            for name, losparse_linear in layer.named_modules():
                if isinstance(losparse_linear, LoSparseLinear):
                    log.info(f"Pruning {name}, set weights to zeros")
                    if self.prune_type == PruningType.UNSTRUCTURED:
                        _, pruned_weights = unstructured_magnitude_prune_(
                            losparse_linear.weight.data,
                            metric_function_or_scores=torch.rand_like,
                            sparsity_ratio=self.sparsity_ratio,
                            return_pruned_weight=True,
                        )
                    elif self.prune_type == PruningType.SEMISTRUCTURED:
                        _, pruned_weights = semistructured_magnitude_prune_(
                            losparse_linear.weight.data,
                            metric_function_or_scores=torch.rand_like,
                            n=self.n,
                            m=self.m,
                            return_pruned_weight=True,
                        )
                    else:
                        raise ValueError(f"Invalid pruning type: {self.prune_type}")
                    self.check_sparsity(losparse_linear.weight)
                    self.extract_low_rank_part_using_pruned_(
                        losparse_linear, pruned_weights
                    )

    def magnitude_prune_(self, model):
        layers: nn.ModuleList = model.model.layers
        for layer_idx, layer in tqdm(
            enumerate(layers), "Pruning Layers", total=len(layers), dynamic_ncols=True
        ):
            for name, losparse_linear in layer.named_modules():
                if isinstance(losparse_linear, LoSparseLinear):
                    log.info(f"Magnitude Pruning {name}")
                    if self.prune_type == PruningType.UNSTRUCTURED:
                        _, pruned_weights = unstructured_magnitude_prune_(
                            losparse_linear.weight.data,
                            metric_function_or_scores=torch.abs,
                            sparsity_ratio=self.sparsity_ratio,
                            return_pruned_weight=True,
                        )
                    elif self.prune_type == PruningType.SEMISTRUCTURED:
                        _, pruned_weights = semistructured_magnitude_prune_(
                            losparse_linear.weight.data,
                            metric_function_or_scores=torch.abs,
                            n=self.n,
                            m=self.m,
                            return_pruned_weight=True,
                        )
                    else:
                        raise ValueError(f"Invalid pruning type: {self.prune_type}")
                    self.check_sparsity(losparse_linear.weight)
                    self.extract_low_rank_part_using_pruned_(
                        losparse_linear, pruned_weights
                    )

    def prune_using_calibration_data_(
        self,
        model: LoSparseLlamaForCausalLM,
        *,
        inps: Tensor,
        outs: Tensor,
        attention_mask: Optional[Tensor],
        position_ids: Optional[Tensor],
    ):
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
            ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = model.hf_device_map[f"model.layers.{layer_idx}"]
                inps, outs, attention_mask, position_ids = (
                    inps.to(dev),
                    outs.to(dev),
                    attention_mask.to(dev) if attention_mask is not None else None,
                    position_ids.to(dev) if position_ids is not None else None,
                )

            # collect the importance scores
            linear_layers = cast(
                Dict[str, LoSparseLinear],
                find_linear_layers(layer, layers=[LoSparseLinear]),
            )

            # register hooks to collect the importance scores
            def get_hook_fn(linear: LoSparseLinear):
                hook_fn = self._variants_hook_mapping[self.variant](linear)
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
            pruned_weights_dict = {}
            for name, linear in linear_layers.items():
                log.info(f"Pruning {name}")
                if self.prune_type == PruningType.UNSTRUCTURED:
                    _, pruned_weights = unstructured_magnitude_prune_(
                        linear.weight.data,
                        metrics[name],
                        sparsity_ratio=self.sparsity_ratio,
                        return_pruned_weight=True,
                    )
                elif self.prune_type == PruningType.SEMISTRUCTURED:
                    _, pruned_weights = semistructured_magnitude_prune_(
                        linear.weight.data,
                        metrics[name],
                        n=self.n,
                        m=self.m,
                        return_pruned_weight=True,
                    )
                else:
                    raise ValueError(f"Invalid pruning type: {self.prune_type}")
                self.check_sparsity(linear.weight)
                pruned_weights_dict[name] = pruned_weights

            # compute the input to the next layer
            with torch.no_grad():
                for j in range(self.nsamples):
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
            inps, outs = outs, inps

            # extract the low-rank parts
            for name, linear in linear_layers.items():
                log.info(f"Extracting low-rank part for {name}")
                self.extract_low_rank_part_using_pruned_(
                    linear, pruned_weights_dict[name]
                )
                linear.skip_lowrank = False

    @torch.no_grad()
    def extract_low_rank_part_using_pruned_(
        self, linear: LoSparseLinear, pruned_weight: Tensor
    ):
        assert isinstance(
            linear, LoSparseLinear
        ), f"Expected LoSparseLinear, got {type(linear)}"

        u, s, vh = cast(
            Tuple[Tensor, Tensor, Tensor],
            torch.linalg.svd(pruned_weight.float(), full_matrices=False),
        )
        v = vh.T
        uk = u[:, : self.rank]
        sk = s[: self.rank]
        vk = v[:, : self.rank]
        linear.lo_A.data = vk.T.to(linear.lo_A.dtype).contiguous()
        linear.lo_B.data = (uk * sk).to(linear.lo_B.dtype).contiguous()
        return linear

    @torch.no_grad()
    def check_sparsity(self, weight: Tensor, tol: float = 0.01):
        if self.prune_type == PruningType.UNSTRUCTURED:
            assert (compute_sparsity(weight) - self.sparsity_ratio).abs() < tol
        elif self.prune_type == PruningType.SEMISTRUCTURED:
            assert (compute_sparsity(weight) - self.n / self.m).abs() < tol
        else:
            raise ValueError(f"Invalid pruning type: {self.prune_type}")


class PCPSparseLoForLlama(SparseLoForLlama):
    "PCP with mask"

    _config_mapping = SparseLoForLlama._config_mapping | {
        "num_iterations": "num_iterations",
    }

    def __init__(self, num_iterations: int, **kwargs):
        super().__init__(**kwargs)
        self.num_iterations = num_iterations

    @override
    def run(self, modelpool):
        if self.seed is not None:
            L.seed_everything(self.seed)

        # load pre-trained model or the first model in the pool
        with self.profile("load_model"):
            model = modelpool.load_pretrained_or_first_model()
            model.seqlen = model.config.max_position_embeddings
            tokenizer = modelpool.load_tokenizer(use_fast=False)

        if not isinstance(model, (LlamaForCausalLM,)):
            log.warning(f"Model type {type(model)} may not supported.")

        if self.variant in self._variants_requires_calibration_data:
            inps, outs, attention_mask, position_ids = self.prepare_calibration_data(
                model, tokenizer
            )

        model = convert_to_losparse_llama(model, rank=self.rank)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for linear in find_linear_layers(model, layers=[LoSparseLinear]).values():
            linear = cast(LoSparseLinear, linear)
            linear.lo_A.data.zero_()
            linear.lo_B.data.zero_()
            linear.skip_lowrank = True

        match self.variant:
            case "dense":
                # this variant is a no-op, just for debug the conversion
                pass
            case "lowrank-only":
                self.extract_low_rank_parts_(model)
                self.set_weights_to_zeros_(model)
            case "random":
                self.pcp_random_prune_(model)
            case "magnitude":
                self.pcp_magnitude_prune_(model)
            case variant if variant in self._variants_requires_calibration_data:
                self.pcp_prune_using_calibration_data_(
                    model,
                    inps=inps,
                    outs=outs,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
            case _:
                raise ValueError(f"Invalid variant: {self.variant}")

        if self.model_save_path is not None:
            with timeit_context(f"Saving the model to {self.model_save_path}"):
                tokenizer.save_pretrained(self.model_save_path)
                model.save_pretrained(self.model_save_path)

        return model

    @torch.no_grad()
    def pcp_random_prune_(self, model):
        layers: nn.ModuleList = model.model.layers
        for layer_idx, layer in tqdm(
            list(enumerate(layers)),
            "Pruning Layers",
            dynamic_ncols=True,
        ):
            for name, linear in layer.named_modules():
                if isinstance(linear, LoSparseLinear):
                    log.info(f"Pruning {name}, set weights to zeros")
                    W = linear.weight.data.clone()
                    if self.prune_type == PruningType.UNSTRUCTURED:
                        unstructured_magnitude_prune_(
                            linear.weight.data,
                            metric_function_or_scores=torch.rand_like,
                            sparsity_ratio=self.sparsity_ratio,
                        )
                    elif self.prune_type == PruningType.SEMISTRUCTURED:
                        semistructured_magnitude_prune_(
                            linear.weight.data,
                            metric_function_or_scores=torch.rand_like,
                            n=self.n,
                            m=self.m,
                        )
                    else:
                        raise ValueError(f"Invalid pruning type: {self.prune_type}")
                    self.check_sparsity(linear.weight)
                    mask = linear.weight != 0
                    linear.weight.data = PCP_search_with_mask(
                        W, mask, T_max=self.num_iterations
                    )
                    self.extract_low_rank_part_using_pruned_(linear, W - linear.weight)

    def pcp_magnitude_prune_(self, model):
        layers: nn.ModuleList = model.model.layers
        for layer_idx, layer in tqdm(
            enumerate(layers), "Pruning Layers", total=len(layers), dynamic_ncols=True
        ):
            for name, linear in layer.named_modules():
                if isinstance(linear, LoSparseLinear):
                    log.info(f"Magnitude Pruning {name}")
                    W = linear.weight.data.clone()
                    if self.prune_type == PruningType.UNSTRUCTURED:
                        unstructured_magnitude_prune_(
                            linear.weight.data,
                            metric_function_or_scores=torch.abs,
                            sparsity_ratio=self.sparsity_ratio,
                        )
                    elif self.prune_type == PruningType.SEMISTRUCTURED:
                        semistructured_magnitude_prune_(
                            linear.weight.data,
                            metric_function_or_scores=torch.abs,
                            n=self.n,
                            m=self.m,
                        )
                    else:
                        raise ValueError(f"Invalid pruning type: {self.prune_type}")
                    self.check_sparsity(linear.weight)
                    mask = linear.weight != 0
                    linear.weight.data = PCP_search_with_mask(
                        W, mask, T_max=self.num_iterations
                    )
                    self.extract_low_rank_part_using_pruned_(linear, W - linear.weight)

    def pcp_prune_using_calibration_data_(
        self,
        model: LoSparseLlamaForCausalLM,
        *,
        inps: Tensor,
        outs: Tensor,
        attention_mask: Optional[Tensor],
        position_ids: Optional[Tensor],
    ):
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
            ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = model.hf_device_map[f"model.layers.{layer_idx}"]
                inps, outs, attention_mask, position_ids = (
                    inps.to(dev),
                    outs.to(dev),
                    attention_mask.to(dev) if attention_mask is not None else None,
                    position_ids.to(dev) if position_ids is not None else None,
                )

            # collect the importance scores
            linear_layers = cast(
                Dict[str, LoSparseLinear],
                find_linear_layers(layer, layers=[LoSparseLinear]),
            )

            # register hooks to collect the importance scores
            def get_hook_fn(linear: LoSparseLinear):
                hook_fn = self._variants_hook_mapping[self.variant](linear)
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
            for name, linear in linear_layers.items():
                log.info(f"Pruning {name}")
                W = linear.weight.data.clone()
                if self.prune_type == PruningType.UNSTRUCTURED:
                    _, pruned_weights = unstructured_magnitude_prune_(
                        linear.weight.data,
                        metrics[name],
                        sparsity_ratio=self.sparsity_ratio,
                        return_pruned_weight=True,
                    )
                elif self.prune_type == PruningType.SEMISTRUCTURED:
                    _, pruned_weights = semistructured_magnitude_prune_(
                        linear.weight.data,
                        metrics[name],
                        n=self.n,
                        m=self.m,
                        return_pruned_weight=True,
                    )
                else:
                    raise ValueError(f"Invalid pruning type: {self.prune_type}")
                self.check_sparsity(linear.weight)
                mask = linear.weight != 0
                linear.weight.data = PCP_search_with_mask(
                    W, mask, T_max=self.num_iterations
                )
                self.extract_low_rank_part_using_pruned_(linear, W - linear.weight)
                linear.skip_lowrank = False

            # compute the input to the next layer
            with torch.no_grad():
                for j in range(self.nsamples):
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
            inps, outs = outs, inps


class IterativeSparseLoForLlama(SparseLoForLlama):
    "Iterative Weight Update"

    _config_mapping = SparseLoForLlama._config_mapping | {
        "num_iterations": "num_iterations",
    }

    def __init__(
        self, num_iterations: int, use_reference_model: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_iterations = num_iterations
        self.use_reference_model = use_reference_model

    @override
    def run(self, modelpool):
        self.modelpool = modelpool
        if self.seed is not None:
            L.seed_everything(self.seed)

        # load pre-trained model or the first model in the pool
        with self.profile("load_model"):
            model = modelpool.load_pretrained_or_first_model()
            model.seqlen = model.config.max_position_embeddings
            tokenizer = modelpool.load_tokenizer(use_fast=False)

        if not isinstance(model, (LlamaForCausalLM,)):
            log.warning(f"Model type {type(model)} may not supported.")

        if self.variant in self._variants_requires_calibration_data:
            inps, outs, attention_mask, position_ids = self.prepare_calibration_data(
                model, tokenizer
            )

        model = convert_to_losparse_llama(model, rank=self.rank)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for linear in find_linear_layers(model, layers=[LoSparseLinear]).values():
            linear = cast(LoSparseLinear, linear)
            linear.lo_A.data.zero_()
            linear.lo_B.data.zero_()
            linear.skip_lowrank = True

        match self.variant:
            case "dense":
                # this variant is a no-op, just for debug the conversion
                pass
            case "lowrank-only":
                self.extract_low_rank_parts_(model)
                self.set_weights_to_zeros_(model)
            case "random":
                self.iterative_random_prune_(model)
            case "magnitude":
                self.iterative_magnitude_prune_(model)
            case variant if variant in self._variants_requires_calibration_data:
                self.iterative_prune_using_calibration_data_(
                    model,
                    inps=inps,
                    outs=outs,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
            case _:
                raise ValueError(f"Invalid variant: {self.variant}")

        if self.model_save_path is not None:
            with timeit_context(f"Saving the model to {self.model_save_path}"):
                tokenizer.save_pretrained(self.model_save_path)
                model.save_pretrained(self.model_save_path)

        return model

    @torch.no_grad()
    def iterative_random_prune_(self, model):
        layers: nn.ModuleList = model.model.layers
        for layer_idx, layer in tqdm(
            list(enumerate(layers)),
            "Pruning Layers",
            dynamic_ncols=True,
        ):
            for name, linear in layer.named_modules():
                if isinstance(linear, LoSparseLinear):
                    log.info(f"Pruning {name}, set weights to zeros")
                    W = linear.weight.data.clone()
                    if self.prune_type == PruningType.UNSTRUCTURED:
                        unstructured_magnitude_prune_(
                            linear.weight.data,
                            metric_function_or_scores=torch.rand_like,
                            sparsity_ratio=self.sparsity_ratio,
                        )
                    elif self.prune_type == PruningType.SEMISTRUCTURED:
                        semistructured_magnitude_prune_(
                            linear.weight.data,
                            metric_function_or_scores=torch.rand_like,
                            n=self.n,
                            m=self.m,
                        )
                    else:
                        raise ValueError(f"Invalid pruning type: {self.prune_type}")
                    self.check_sparsity(linear.weight)
                    mask = linear.weight != 0
                    for rank in tqdm(
                        np.linspace(1, self.rank, self.num_iterations, dtype=np.int64),
                        "Iterative Pruning",
                        leave=False,
                        dynamic_ncols=True,
                    ):
                        linear.weight.data, specturm_ratio = iterative_weight_update(
                            W,
                            linear.weight,
                            mask,
                            rank=rank,
                        )
                        if specturm_ratio > 0.99:
                            break
                    self.extract_low_rank_part_using_pruned_(linear, W - linear.weight)

    @torch.no_grad()
    def iterative_magnitude_prune_(self, model):
        layers: nn.ModuleList = model.model.layers
        if self.use_reference_model:
            reference_model = self.modelpool.load_model(
                "reference_model", torch_dtype="float16"
            )
            reference_layers: nn.ModuleList = reference_model.model.layers
        for layer_idx, layer in tqdm(
            enumerate(layers), "Pruning Layers", total=len(layers), dynamic_ncols=True
        ):
            for name, linear in layer.named_modules():
                if isinstance(linear, LoSparseLinear):
                    log.info(f"Magnitude Pruning {name}")
                    W = (
                        linear.weight.data.clone()
                        if not self.use_reference_model
                        else reference_layers[layer_idx]
                        .get_submodule(name)
                        .weight.data.clone()
                        .to(linear.weight.data.device)
                    )
                    if self.prune_type == PruningType.UNSTRUCTURED:
                        unstructured_magnitude_prune_(
                            linear.weight.data,
                            metric_function_or_scores=torch.abs,
                            sparsity_ratio=self.sparsity_ratio,
                        )
                    elif self.prune_type == PruningType.SEMISTRUCTURED:
                        semistructured_magnitude_prune_(
                            linear.weight.data,
                            metric_function_or_scores=torch.abs,
                            n=self.n,
                            m=self.m,
                        )
                    else:
                        raise ValueError(f"Invalid pruning type: {self.prune_type}")
                    self.check_sparsity(linear.weight)
                    mask = linear.weight != 0
                    for rank in tqdm(
                        np.linspace(1, self.rank, self.num_iterations, dtype=np.int64),
                        "Iterative Pruning",
                        leave=False,
                        dynamic_ncols=True,
                    ):
                        linear.weight.data, specturm_ratio = iterative_weight_update(
                            W,
                            linear.weight,
                            mask,
                            rank=rank,
                        )
                        if specturm_ratio > 0.99:
                            break
                    self.extract_low_rank_part_using_pruned_(linear, W - linear.weight)

    @torch.no_grad()
    def iterative_prune_using_calibration_data_(
        self,
        model: LoSparseLlamaForCausalLM,
        *,
        inps: Tensor,
        outs: Tensor,
        attention_mask: Optional[Tensor],
        position_ids: Optional[Tensor],
    ):
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
            ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = model.hf_device_map[f"model.layers.{layer_idx}"]
                inps, outs, attention_mask, position_ids = (
                    inps.to(dev),
                    outs.to(dev),
                    attention_mask.to(dev) if attention_mask is not None else None,
                    position_ids.to(dev) if position_ids is not None else None,
                )

            # collect the importance scores
            linear_layers = cast(
                Dict[str, LoSparseLinear],
                find_linear_layers(layer, layers=[LoSparseLinear]),
            )

            # register hooks to collect the importance scores
            def get_hook_fn(linear: LoSparseLinear):
                hook_fn = self._variants_hook_mapping[self.variant](linear)
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
            for name, linear in linear_layers.items():
                log.info(f"Pruning {name}")
                W = linear.weight.data.clone()
                if self.prune_type == PruningType.UNSTRUCTURED:
                    _, pruned_weights = unstructured_magnitude_prune_(
                        linear.weight.data,
                        metrics[name],
                        sparsity_ratio=self.sparsity_ratio,
                        return_pruned_weight=True,
                    )
                elif self.prune_type == PruningType.SEMISTRUCTURED:
                    _, pruned_weights = semistructured_magnitude_prune_(
                        linear.weight.data,
                        metrics[name],
                        n=self.n,
                        m=self.m,
                        return_pruned_weight=True,
                    )
                else:
                    raise ValueError(f"Invalid pruning type: {self.prune_type}")
                self.check_sparsity(linear.weight)
                mask = linear.weight != 0
                for rank in tqdm(
                    np.linspace(1, self.rank, self.num_iterations, dtype=np.int64),
                    "Iterative Pruning",
                    leave=False,
                    dynamic_ncols=True,
                ):
                    linear.weight.data, specturm_ratio = iterative_weight_update(
                        W,
                        linear.weight,
                        mask,
                        rank=rank,
                    )
                    if specturm_ratio > 0.99:
                        break
                self.extract_low_rank_part_using_pruned_(linear, W - linear.weight)
                linear.skip_lowrank = False

            # compute the input to the next layer
            with torch.no_grad():
                for j in range(self.nsamples):
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
            inps, outs = outs, inps
