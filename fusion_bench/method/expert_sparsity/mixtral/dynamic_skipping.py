R"""
Example:

```bash
fusion_bench \
    fabric.loggers.name="mixtral_8x7b_expert_pruning/dynamic_skipping" \
    method=expert_sparsity/mixtral \
    method._target_=fusion_bench.method.DynamicSkippingPruningForMixtral \
    modelpool=CausalLMPool/mixtral-8x7b
```
"""

import logging
import os

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MixtralForCausalLM
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM

import fusion_bench as fb
from fusion_bench.method.expert_sparsity.utils.calibration_data import (
    build_calib_loader,
)
from fusion_bench.models.expert_sparsity.mixtral.wrapper import (
    PrunableMixtralSparseMoeBlockWrapper,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def dynamic_skipping(
    model: MixtralForCausalLM,
    calib_loader: DataLoader,
    batch_size: int,
):
    assert isinstance(
        model, MixtralForCausalLM
    ), "Currently only `Mixtral` is supported"

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = PrunableMixtralSparseMoeBlockWrapper(
            layer.block_sparse_moe
        )
        layer.block_sparse_moe.cache_logits = True
        layer.block_sparse_moe.cache_X = True
        layer.block_sparse_moe.cache_Z = True

    with torch.inference_mode():
        for i, batch in enumerate(
            tqdm(calib_loader, desc="Model forwarding on sample set...")
        ):
            model_inputs = model.prepare_inputs_for_generation(**batch)
            outputs = model(**model_inputs)
            assert outputs is not None

    res_median = {}
    res_mean = {}

    for layer_idx in range(len(model.model.layers)):
        b = model.model.layers[layer_idx].block_sparse_moe
        b.cache_space.prepare_for_loader()
        dataloader = torch.utils.data.DataLoader(
            b.cache_space,
            batch_size=batch_size,
            shuffle=True,
        )
        logger.info(len(dataloader))

        ana_list = []
        for i, (router_logits, X, Z) in enumerate(dataloader):
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float).view(
                -1, b.model.num_experts
            )
            for j in range(len(routing_weights)):
                sorted_weights, sort_indices = torch.sort(
                    routing_weights[j], descending=True
                )
                ana_list.append(float(sorted_weights[1] / sorted_weights[0]))

        median = np.median(ana_list)
        mean = np.mean(ana_list)
        logger.info(f"layer {layer_idx} | mean: {mean}, median: {median}")
        res_median[str(layer_idx)] = median
        res_mean[str(layer_idx)] = mean

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = layer.block_sparse_moe.model

    model.config.betas = res_median
    return model, (res_median, res_mean)


class DynamicSkippingPruningForMixtral(
    fb.BaseAlgorithm,
    fb.mixins.LightningFabricMixin,
    fb.mixins.SimpleProfilerMixin,
):
    modelpool: fb.modelpool.CausalLMPool

    def __init__(
        self,
        calib_set: str,
        max_block_size: int,
        n_blocks_for_stat: int,
        batch_size: int,
        num_workers: int,
        num_preserved_experts: int,
        seed: int = 42,
        model_save_path: str = R"{log_dir}/pruned_model",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_save_path = model_save_path
        self.calib_set = calib_set
        self.max_block_size = max_block_size
        self.n_blocks_for_stat = n_blocks_for_stat
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.num_preserved_experts = num_preserved_experts

    def run(self, modelpool: fb.modelpool.CausalLMPool):
        """
        Args:
            modelpool (fb.modelpool.CausalLMPool): The model pool to run the algorithm on.
                Example Config: config/modelpool/CausalLMPool/mixtral-8x7b.yaml
        """
        self.modelpool = modelpool
        # set random seed
        if self.seed is not None:
            L.seed_everything(self.seed)
        # parse model_save_path
        self.model_save_path = self.model_save_path.format(log_dir=self.log_dir)

        with self.profile("load model"):
            model = modelpool.load_pretrained_or_first_model()
            tokenizer = modelpool.load_tokenizer()

        # Load the calibration data
        with self.profile("load calibration data"):
            calib_loader = build_calib_loader(
                self.calib_set,
                tokenizer=tokenizer,
                max_block_size=self.max_block_size,
                n_blocks_for_stat=self.n_blocks_for_stat,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                seed=self.seed,
            )

        with self.profile("prune model"):
            model, info = dynamic_skipping(
                model,
                calib_loader,
                batch_size=self.batch_size,
            )

        if self.model_save_path is not None:
            with self.profile("save model"):
                modelpool.save_model(
                    model,
                    path=self.model_save_path,
                    tokenizer=tokenizer,
                )
                torch.save(info, os.path.join(self.log_dir, "pruning_info.pt"))

        self.print_profile_summary()
        return model
