import logging
import os
import re
from copy import deepcopy
from typing import Dict, List, cast

import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import LlamaForCausalLM

from fusion_bench.method import ModelFusionAlgorithm
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import ModelPool, to_modelpool
from fusion_bench.modelpool.huggingface_llm import LLamaForCausalLMPool

from .wanda_utils.prune import llama_prune_wanda_


class WandaPruningForLlama(ModelFusionAlgorithm, SimpleProfilerMixin):

    def run(self, modelpool: LLamaForCausalLMPool):
        config = self.config

        # load pre-trained model or the first model in the pool
        with self.profile("load_model"):
            model = cast(LlamaForCausalLM, modelpool.load_pretrained_or_first_model())
            model.seqlen = model.config.max_position_embeddings
            tokenizer = modelpool.load_pretrained_or_first_tokenizer(use_fast=False)

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
        return model
