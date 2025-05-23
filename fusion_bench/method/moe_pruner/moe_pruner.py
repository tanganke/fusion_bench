import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, TypeVar, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformers import MixtralForCausalLM
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralSparseMoeBlock,
)

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.method.pruning.prune_utils import (
    PruningType,
    compute_sparsity,
    find_linear_layers,
    semistructured_magnitude_prune_,
    unstructured_magnitude_prune_,
)
from fusion_bench.mixins import LightningFabricMixin, SimpleProfilerMixin
from fusion_bench.modelpool import CausalLMPool
from fusion_bench.models.modeling_deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2MLP,
    DeepseekV2MoE,
    DeepseekV2MoEGate,
)
from fusion_bench.utils import timeit_context
from fusion_bench.utils.cache_utils import cache_to_disk
from fusion_bench.utils.devices import to_device

from .hooks.deepseek_v2 import (
    MoEPrunerHookFnForDeepseekV2Gate,
    MoEPrunerHookFnForDeepseekV2Linear,
)
from .hooks.hook import BaseHookFn
from .hooks.mixtral import (
    MoEPrunerHookFnForMixtralGate,
    MoEPrunerHookFnForMixtralLinear,
)
from .utils.data import get_loaders
from .utils.prune import prepare_calibration_input

MoEModel = TypeVar("MoEModel", bound=Union[MixtralForCausalLM, DeepseekV2ForCausalLM])

log = logging.getLogger(__name__)


class MoEPruner(BaseAlgorithm, SimpleProfilerMixin, LightningFabricMixin):

    def __init__(
        self,
        nsamples: int,
        seed: int,
        device: str,
        prune_type: PruningType,
        sparsity_ratio: float,
        n: int,
        m: int,
        max_seqlen: Optional[int] = None,
    ):
        self.nsamples = nsamples
        self.seed = seed
        self.device = device
        self.max_seqlen = max_seqlen
        self.prune_type = prune_type
        self.sparsity_ratio = sparsity_ratio
        self.n = n
        self.m = m
        super().__init__()

    def run(self, modelpool: CausalLMPool):
        # load pre-trained model or the first model in the pool
        with self.profile("load_model"):
            model: MoEModel = modelpool.load_pretrained_or_first_model()
            if self.max_seqlen is not None:
                model.seqlen = min(
                    model.config.max_position_embeddings,
                    self.max_seqlen,
                )
            tokenizer = modelpool.load_tokenizer()

        inps, outs, attention_mask, position_ids, position_embeddings = (
            self.prepare_calibration_data(model, tokenizer)
        )

        self.prune_using_calibration_data_(
            model,
            inps=inps,
            outs=outs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

        return model

    def prepare_calibration_data(self, model: MoEModel, tokenizer):
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
            inps, outs, attention_mask, position_ids, position_embeddings = (
                prepare_calibration_input(model, dataloader, self.device)
            )
        return inps, outs, attention_mask, position_ids, position_embeddings

    def prune_using_calibration_data_(
        self,
        model: MoEModel,
        *,
        inps,
        outs,
        attention_mask,
        position_ids,
        position_embeddings,
    ):
        model.eval()
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
                # handle the case for large models, when the device map has multiple GPUs;
                dev = model.hf_device_map[f"model.layers.{layer_idx}"]
                inps, outs, attention_mask, position_ids, position_embeddings = (
                    inps.to(dev),
                    outs.to(dev),
                    attention_mask.to(dev) if attention_mask is not None else None,
                    position_ids.to(dev) if position_ids is not None else None,
                    (
                        to_device(position_embeddings, dev)
                        if position_embeddings is not None
                        else None
                    ),
                )

            if isinstance(layer, MixtralDecoderLayer):
                linear_layers = find_linear_layers(layer.block_sparse_moe.experts)
            elif isinstance(layer, DeepseekV2DecoderLayer):
                if isinstance(layer.mlp, DeepseekV2MoE):
                    linear_layers = find_linear_layers(layer.mlp.experts)
                elif isinstance(layer.mlp, DeepseekV2MLP):
                    # compute the input to the next layer
                    with torch.no_grad():
                        for j in range(self.nsamples):
                            outs[j] = layer(
                                inps[j].unsqueeze(0),
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                position_embeddings=position_embeddings,
                            )[0]
                    inps, outs = outs, inps
                    continue
            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")

            linear_hooks: Dict[str, BaseHookFn] = {}
            handles: List[torch.utils.hooks.RemovableHandle] = []
            for name, linear in linear_layers.items():
                if isinstance(model, MixtralForCausalLM):
                    hook_fn = MoEPrunerHookFnForMixtralLinear(linear, name)
                elif isinstance(model, DeepseekV2ForCausalLM):
                    hook_fn = MoEPrunerHookFnForDeepseekV2Linear(linear, name)
                else:
                    raise ValueError(f"Unsupported model type: {type(model)}")
                linear_hooks[name] = hook_fn
                handles.append(linear.register_forward_hook(hook_fn))

            if isinstance(model, MixtralForCausalLM):
                gate_hook = MoEPrunerHookFnForMixtralGate(
                    layer.block_sparse_moe.gate,
                    linear_hooks,
                    top_k=layer.block_sparse_moe.top_k,
                    num_experts=layer.block_sparse_moe.num_experts,
                )
                handles.append(
                    layer.block_sparse_moe.gate.register_forward_hook(gate_hook)
                )
            elif isinstance(model, DeepseekV2ForCausalLM):
                gate_hook = MoEPrunerHookFnForDeepseekV2Gate(
                    layer.mlp.gate,
                    linear_hooks,
                    top_k=layer.mlp.gate.top_k,
                    num_experts=layer.mlp.config.n_routed_experts,
                )
                handles.append(layer.mlp.gate.register_forward_hook(gate_hook))
            else:
                raise ValueError(f"Unsupported model type: {type(model)}")

            with torch.no_grad():
                for j in range(self.nsamples):
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )[0]

            # compute the importance scores and remove the hooks
            metrics = {}
            for name, hook in linear_hooks.items():
                metrics[name] = hook.compute().detach().cpu()
            for h in handles:
                h.remove()

            # prune the weights based on the importance scores
            if self.prune_type == PruningType.UNSTRUCTURED:
                for name, linear in linear_layers.items():
                    log.info(f"Pruning {name}")
                    unstructured_magnitude_prune_(
                        linear.weight.data,
                        metrics[name].to(linear.weight.device),
                        sparsity_ratio=self.sparsity_ratio,
                    )
                    self.check_sparsity(linear.weight)
            elif self.prune_type == PruningType.SEMISTRUCTURED:
                for name, linear in linear_layers.items():
                    log.info(f"Pruning {name}")
                    semistructured_magnitude_prune_(
                        linear.weight.data,
                        metrics[name].to(linear.weight.device),
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
                        position_embeddings=position_embeddings,
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
