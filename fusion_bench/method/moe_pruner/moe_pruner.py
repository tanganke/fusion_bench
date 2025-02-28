import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, TypeVar

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
from fusion_bench.utils import timeit_context
from fusion_bench.utils.cache_utils import cache_to_disk
from fusion_bench.utils.devices import to_device

from .utils.data import get_loaders
from .utils.hook import BaseHookFn
from .utils.prune import prepare_calibration_input

MoEModel = TypeVar("MoEModel", bound=MixtralForCausalLM)

log = logging.getLogger(__name__)


class MoEPrunerHookFnForMixtralLinear(BaseHookFn):
    _routing_weights = None  # set by gate hook

    def __init__(
        self,
        linear: nn.Linear,
        name: str,
    ):
        super().__init__(linear)
        self.linear = linear
        self.scalar_row = torch.zeros(
            (linear.weight.size(1),), device=linear.weight.device
        )
        self.nsamples = 0
        self.name = name

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
        routing_weights = self._routing_weights.t()
        self.scalar_row += (
            torch.norm(inp * routing_weights, p=2, dim=1) ** 2 / self.nsamples
        )


class MoEPrunerLinearHookFnForMixtralGate(BaseHookFn):

    def __init__(
        self,
        router: nn.Module,
        linear_layer_hooks: Dict[str, MoEPrunerHookFnForMixtralLinear],
        top_k: int,
        num_experts: int,
    ):
        self.nsamples = 0
        self.linear_layer_hooks = linear_layer_hooks
        self.top_k = top_k
        self.num_experts = num_experts
        super().__init__(router)

    def __call__(self, router, inps: Tuple[Tensor], out: Tensor):
        assert len(inps) == 1
        inp = inps[0]

        router_logits = out
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            for name, hook in self.linear_layer_hooks.items():
                if not name.startswith(f"{expert_idx}."):
                    continue
                hook._routing_weights = routing_weights[top_x, idx, None]

    def compute(self):
        pass


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
            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")

            linear_hooks: Dict[str, BaseHookFn] = {}
            handles: List[torch.utils.hooks.RemovableHandle] = []
            for name, linear in linear_layers.items():
                if isinstance(model, MixtralForCausalLM):
                    hook_fn = MoEPrunerHookFnForMixtralLinear(linear, name)
                else:
                    raise ValueError(f"Unsupported model type: {type(model)}")
                linear_hooks[name] = hook_fn
                handles.append(linear.register_forward_hook(hook_fn))

            if isinstance(model, MixtralForCausalLM):
                gate_hook = MoEPrunerLinearHookFnForMixtralGate(
                    layer.block_sparse_moe.gate,
                    linear_hooks,
                    top_k=layer.block_sparse_moe.top_k,
                    num_experts=layer.block_sparse_moe.num_experts,
                )
                handles.append(
                    layer.block_sparse_moe.gate.register_forward_hook(gate_hook)
                )
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
