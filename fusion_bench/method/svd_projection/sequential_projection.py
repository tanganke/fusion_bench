import random
from typing import TYPE_CHECKING, List, Tuple, cast

import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from tqdm.auto import tqdm

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.method import SimpleAverageAlgorithm, TaskArithmeticAlgorithm
from fusion_bench.utils import instantiate

from .utils import frobenius_inner_product, svd


class SequentialProjection(BaseAlgorithm):
    def __init__(
        self,
        scaling_factor: float,
        alpha: float,
        base_algorithm: DictConfig,
        shuffle_order: bool = True,
        accelerator: str = "auto",
        **kwargs,
    ):
        self.scaling_factor = scaling_factor
        self.alpha = alpha
        self.shuffle_order = shuffle_order
        self._base_algorithm = base_algorithm
        self.accelerator = accelerator
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        if self.accelerator == "auto":
            accelerator = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            accelerator = self.accelerator

        pretrained_model = modelpool.load_pretrained_model()
        model_names = modelpool.model_names
        if self.shuffle_order:
            random.shuffle(model_names)

        # get the average model
        merged_model = instantiate(self._base_algorithm).run(modelpool)
        # copy the first model to the merged model
        first_model = modelpool.load_model(model_names[0])
        for name, module in first_model.named_modules():
            if isinstance(module, nn.Linear):
                # merged_model.get_submodule(name).weight.data = module.weight.data
                merged_model.get_submodule(name).weight.data = (
                    self.scaling_factor
                    * (
                        module.weight.data
                        - pretrained_model.get_submodule(name).weight.data
                    )
                    + pretrained_model.get_submodule(name).weight.data
                )

        for model_name in tqdm(model_names[1:], desc="Processing models"):
            task_model = modelpool.load_model(model_name)
            for name, module in tqdm(
                list(merged_model.named_modules()),
                desc=f"Processing {model_name}",
                leave=False,
            ):
                if isinstance(module, nn.Linear):
                    merged_weight = module.weight.data
                    pretrained_weight = pretrained_model.get_submodule(name).weight.data
                    task_weight = task_model.get_submodule(name).weight.data

                    original_device = merged_weight.device
                    merged_weight = merged_weight.to(accelerator)
                    pretrained_weight = pretrained_weight.to(accelerator)
                    task_weight = task_weight.to(accelerator)

                    # skip if the weight is the same as the pretrained model
                    if torch.eq(merged_weight, pretrained_weight).all():
                        continue

                    merged_tv = merged_weight - pretrained_weight
                    task_tv = task_weight - pretrained_weight  # * self.scaling_factor

                    u, s, v = svd(merged_tv, accelerator=accelerator)
                    rank = s.size(0)
                    # find the index that sum up to alpha * sum_of_singular_values
                    split_rank = (
                        (s.cumsum(dim=0) / s.sum() > self.alpha).float().argmax().item()
                    )

                    trust_u = u[:, :split_rank]
                    trust_v = v[:, :split_rank]

                    trust_tv = (
                        task_tv - trust_u @ trust_u.T @ task_tv @ trust_v @ trust_v.T
                    )
                    # module.weight.data = (trust_tv + merged_weight).to(original_device)
                    merged_tv_trust = (
                        trust_u @ trust_u.T @ merged_tv @ trust_v @ trust_v.T
                    )
                    merged_tv_untrust = merged_tv - merged_tv_trust
                    module.weight.data = (
                        0.3 * trust_tv + merged_weight#  - merged_tv_untrust
                    ).to(original_device)

        return merged_model
