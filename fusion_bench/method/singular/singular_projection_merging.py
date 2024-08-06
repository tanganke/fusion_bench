import logging
import os
import re
from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from fusion_bench.method import ModelFusionAlgorithm
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import ModelPool, to_modelpool
from fusion_bench.models.utils import get_attr, set_attr

log = logging.getLogger(__name__)


def svd(w: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    u, s, vh = torch.linalg.svd(
        w, full_matrices=True, driver="gesvd" if w.is_cuda else None
    )
    v = vh.T
    return u, s, v


def _is_name_matched(name: str, extract_names: List[str]):
    for extract_name in extract_names:
        # extract_name is a regular expression
        if re.match(extract_name, name):
            return True
    return False


def _total_parameters(state):
    if isinstance(state, Tensor):
        return state.numel()
    elif isinstance(state, dict):
        return sum(_total_parameters(v) for v in state.values())
    else:
        raise ValueError(f"Unsupported type: {type(state)}")


class SingularProjectionMergingAlgorithm(ModelFusionAlgorithm, SimpleProfilerMixin):
    @torch.no_grad()
    def run(self, modelpool: ModelPool):
        """
        Project the parameter differences into pre-trained SVD subspace.
        This is an experimental method to investigate the location of task-specific knowledge.
        """
        modelpool = to_modelpool(modelpool)

        if self.config.model_path is not None and os.path.exists(
            self.config.model_path
        ):
            log.info(f"loading merged model from {self.config.model_path}")
            model = torch.load(self.config.model_path)

        with self.profile("load pretrained model"):
            pretrained_model = modelpool.load_model("_pretrained_")
        with self.profile("load fine-tuned model"):
            finetuned_models = modelpool.load_model(modelpool.model_names[0])

        with self.profile("merge model"):
            model = self.merge(pretrained_model, finetuned_models)

        if self.config.model_path is not None:
            os.path.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            torch.save(model, self.config.model_path)

        self.print_profile_summary()
        return model

    def merge(
        self,
        pretrained_model: nn.Module,
        finetuned_model: nn.Module,
        in_place: bool = True,
    ):
        if in_place:
            model = finetuned_model
        else:
            model = deepcopy(finetuned_model)

        for name, module in tqdm(
            tuple(model.named_modules()),
            "Projection merging in SVD subspace of pretrained model",
        ):
            if isinstance(module, nn.Linear):
                name_list = name.split(".")
                set_attr(
                    model,
                    name_list,
                    self.projection_merge_linear(
                        get_attr(pretrained_model, name_list),
                        get_attr(finetuned_model, name_list),
                        k=self.config.k,
                    ),
                )
        return model

    def projection_merge_linear(
        self, pretrained_model: nn.Linear, finetuned_model: nn.Linear, k: int
    ):
        if k < 0:
            return deepcopy(finetuned_model)

        w = pretrained_model.weight
        w_ft = finetuned_model.weight

        u, s, v = svd(w)
        if self.config.rank == "low":
            u = u[:, :k]
            s = s[:k]
            v = v[:, :k]
        else:
            u = u[:, k:]
            s = s[k:]
            v = v[:, k:]

        w_diff = w_ft - w
        w_diff_proj = u.T @ w_diff @ v
        w.data = w + u @ w_diff_proj @ v.T
        if pretrained_model.bias is not None:
            pretrained_model.bias.data = finetuned_model.bias.data
        return pretrained_model
