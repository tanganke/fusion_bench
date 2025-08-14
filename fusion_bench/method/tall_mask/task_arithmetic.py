"""
Modified from https://github.com/Zhou-Hangyu/randes/tree/main/benchmark/fusion_bench
"""

import logging
from collections import OrderedDict
from copy import deepcopy

import torch

from fusion_bench import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_binary_mask,
    state_dict_diff_abs,
    state_dict_hadmard_product,
    state_dict_mul,
    state_dict_sub,
    state_dict_sum,
)

log = logging.getLogger(__name__)


def generate_task_masks(
    multi_task_vector: OrderedDict,
    ft_task_vector: OrderedDict,
    pretrained_task_vector: OrderedDict,
    tall_mask_lambda: float = 1.0,
) -> OrderedDict:
    """Adopted from https://github.com/nik-dim/tall_masks/tree/master.
    Generate task-specific TALL masks
    TALL masks are generated as: mask_t = |theta_0 - theta_t| > |theta_mt - theta_t| * lambda

    Args:
        multi_task_vector: multi-task vector
        ft_task_vector: individual theta_t (fine-tuned weights)
        pretrained_task_vector: theta_0 (pre-trained weight)
        tall_mask_lambda: hyper-parameter lambda for generating TALL masks
    Returns:
        final_mask: generated TALL masks with the given lambda
    """

    print(f"Generating TALL masks.")

    # generate masks by comparing the l1 distance between |theta_0 - theta_t| and |theta_mt - theta_t|
    diff_pt_ft = state_dict_diff_abs(pretrained_task_vector, ft_task_vector)
    diff_multi_ft = state_dict_diff_abs(multi_task_vector, ft_task_vector)
    # compare the l1 distance, scaled with hyper-parameter lambda
    final_mask = state_dict_binary_mask(
        diff_pt_ft,
        state_dict_mul(diff_multi_ft, tall_mask_lambda),
    )
    for key, value in final_mask.items():
        final_mask[key] = value.float()
    return final_mask


class TallMaskTaskArithmeticAlgorithm(
    BaseAlgorithm,
    SimpleProfilerMixin,
):
    _config_mapping = BaseAlgorithm._config_mapping | {
        "tall_mask_lambda": "tall_mask_lambda",
        "debug": "debug",
        "verbose": "verbose",
    }

    def __init__(
        self,
        tall_mask_lambda: float,
        debug: int = 0,
        verbose: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tall_mask_lambda = tall_mask_lambda
        self.debug = debug
        self.verbose = verbose

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(models=modelpool)

        log.info("Compressing models using tall mask task arithmetic.")
        task_vector = None
        with self.profile("load model"):
            pretrained_model = modelpool.load_model("_pretrained_")

        task_vectors = {}
        models = {}
        for model_name in modelpool.model_names:
            with self.profile("load model"):
                model = modelpool.load_model(model_name)
            for layer_name, layer in model.state_dict(keep_vars=True).items():
                if self.verbose >= 1:
                    log.info(f"{layer_name} | {layer.shape}")
            task_vector = state_dict_sub(
                model.state_dict(keep_vars=True),
                pretrained_model.state_dict(keep_vars=True),
            )
            task_vectors[model_name] = task_vector

        multi_task_vector = state_dict_sum(list(task_vectors.values()))

        tall_masks = {model: {} for model in modelpool.model_names}

        for model_name in modelpool.model_names:
            tall_mask = generate_task_masks(
                multi_task_vector,
                task_vectors[model_name],
                pretrained_model.state_dict(keep_vars=True),
                tall_mask_lambda=self.tall_mask_lambda,
            )
            tall_masks[model_name] = tall_mask

        with self.profile("compress and retrieve"):
            for model_name in modelpool.model_names:
                retrieved_task_vector = state_dict_hadmard_product(
                    tall_masks[model_name], multi_task_vector
                )
                retrieved_state_dict = state_dict_add(
                    pretrained_model.state_dict(keep_vars=True), retrieved_task_vector
                )
                retrieved_model = deepcopy(pretrained_model)
                retrieved_model.load_state_dict(retrieved_state_dict)
                models[model_name] = retrieved_model

        self.print_profile_summary()
        return {"models": models, "metadata": None}
