import logging
import os
import re
from copy import deepcopy
from typing import Dict, List, Literal, Optional, cast

import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import LlamaForCausalLM

from fusion_bench.method import BaseModelFusionAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool, CausalLMPool
from fusion_bench.utils import timeit_context

from .wanda_utils.prune import llama_prune_wanda_


class WandaPruningForLlama(BaseModelFusionAlgorithm, SimpleProfilerMixin):
    _config_mapping = BaseModelFusionAlgorithm._config_mapping | {
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
        prune_type: Literal["unstructured", "semistructured"],
        device: str,
        dtype: str,
        sparsity_ratio: float,
        n: int,
        m: int,
        model_save_path: Optional[str] = None,
        **kwargs,
    ):
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
        config = self.config

        # load pre-trained model or the first model in the pool
        with self.profile("load_model"):
            model = modelpool.load_pretrained_or_first_model()
            model.seqlen = model.config.max_position_embeddings
            tokenizer = modelpool.load_tokenizer(use_fast=False)

        args = DictConfig(
            {
                "nsamples": config.nsamples,
                "seed": config.seed,
            }
        )
        if config.prune_type == "unstructured":
            prune_n = prune_m = 0
            args.sparsity_ratio = config.sparsity_ratio
            args.use_variant = config.use_variant
        elif config.prune_type == "semistructured":
            prune_n, prune_m = config.n, config.m
        else:
            raise ValueError(f"Invalid pruning type: {config.prune_type}")

        llama_prune_wanda_(
            args,
            model=model,
            tokenizer=tokenizer,
            device=torch.device(config.device),
            prune_n=prune_n,
            prune_m=prune_m,
        )

        if self.model_save_path is not None:
            with timeit_context(f"Saving pruned model to {self.model_save_path}"):
                tokenizer.save_pretrained(self.model_save_path)
                model.save_pretrained(self.model_save_path)
        return model
