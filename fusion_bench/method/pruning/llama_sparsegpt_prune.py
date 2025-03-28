import logging
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import LlamaForCausalLM

from fusion_bench import BaseAlgorithm
from fusion_bench.method.pruning.prune_utils import (
    PruningType,
    compute_sparsity,
    find_linear_layers,
    semistructured_magnitude_prune_,
    unstructured_magnitude_prune_,
)
from fusion_bench.method.pruning.sparsegpt_utils import SparseGPT
from fusion_bench.method.pruning.wanda_utils.data import get_loaders
from fusion_bench.method.pruning.wanda_utils.prune import prepare_calibration_input
from fusion_bench.mixins import SimpleProfilerMixin
from fusion_bench.modelpool import CausalLMPool
from fusion_bench.utils import timeit_context
from fusion_bench.utils.cache_utils import cache_to_disk

log = logging.getLogger(__name__)


class SparseGPTPruningForLlama(BaseAlgorithm, SimpleProfilerMixin):
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
        Initialize the SparseGPTPruningForLlama class.

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

    @torch.no_grad()
    def prune_using_calibration_data_(
        self,
        model: LlamaForCausalLM,
        *,
        inps,
        outs,
        attention_mask,
        position_ids,
    ):
        layers = model.model.layers
        for layer_indx, layer in tqdm(
            enumerate(layers),
            "Pruning Layers",
            total=len(layers),
            dynamic_ncols=True,
        ):
            layer = layers[layer_indx]
            if f"model.layers.{layer_indx}" in model.hf_device_map:
                dev = model.hf_device_map[f"model.layers.{layer_indx}"]
                print(f"layer {layer_indx} device {dev}")
                inps, outs, attention_mask, position_ids = (
                    inps.to(dev),
                    outs.to(dev),
                    attention_mask.to(dev),
                    position_ids.to(dev),
                )

            subset = find_linear_layers(layer, layers=[nn.Linear])

            gpts: Dict[str, SparseGPT] = {}
            for name in subset:
                gpts[name] = SparseGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in gpts:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(self.nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
            for h in handles:
                h.remove()

            for name in gpts:
                print(layer_indx, name)
                print("Pruning ...")

                gpts[name].fasterprune(
                    self.sparsity_ratio,
                    prune_n=self.n,
                    prune_m=self.m,
                    percdamp=0.01,
                    blocksize=128,
                )
                gpts[name].free()

            for j in range(self.nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

            layers[layer_indx] = layer
            torch.cuda.empty_cache()

            inps, outs = outs, inps
