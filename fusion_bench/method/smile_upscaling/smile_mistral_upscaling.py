import logging
import os
import re
from copy import deepcopy
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, MistralForCausalLM
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from fusion_bench.compat.method import ModelFusionAlgorithm
from fusion_bench.compat.modelpool import to_modelpool
from fusion_bench.method.simple_average import simple_average
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.models.modeling_smile_mistral import (
    SmileMistralConfig,
    SmileMistralForCausalLM,
)
from fusion_bench.models.modeling_smile_mistral.modeling_smile_mistral import (
    SmileLinear,
    SmileMistralDecoderLayer,
)
from fusion_bench.models.utils import get_attr, set_attr
from fusion_bench.utils.dtype import parse_dtype
from fusion_bench.utils.parameters import print_parameters
from fusion_bench.utils.state_dict_arithmetic import state_dict_sub

log = logging.getLogger(__name__)


class ExpertNotTrainedError(Exception):
    pass


def _is_all_zeros(tensor: Tensor | List[Tensor]) -> bool:
    """
    Check if a tensor or a list of tensors are all zeros.

    Args:
        tensor (Tensor | List[Tensor]): The tensor or list of tensors to check.

    Returns:
        bool: True if all elements are zeros, False otherwise.
    """
    if isinstance(tensor, Tensor):
        return torch.allclose(tensor, torch.zeros_like(tensor))
    else:
        return all(_is_all_zeros(t) for t in tensor)


def _svd(w: Tensor, full_matrices=False) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform Singular Value Decomposition (SVD) on a tensor.

    Args:
        w (Tensor): The input tensor.
        full_matrices (bool, optional): Whether to compute the full-sized U and V matrices. Defaults to False.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The U, S, and V matrices from SVD.
    """
    device = w.device
    if w.device != torch.float32 or w.device != torch.float64:
        w = w.float()

    u, s, vh = torch.linalg.svd(
        w,
        full_matrices=full_matrices,
        # driver="gesvd" if w.is_cuda else None
    )
    v = vh.T

    u = u.to(device)
    s = s.to(device)
    v = v.to(device)
    return u, s, v


def svd(
    w: Tensor, full_matrices=True, accelerator=None
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform SVD on a tensor with optional acceleration.

    Args:
        w (Tensor): The input tensor.
        full_matrices (bool, optional): Whether to compute the full-sized U and V matrices. Defaults to True.
        accelerator (optional): The device to perform the computation on. Defaults to None.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The U, S, and V matrices from SVD.
    """
    if accelerator is None:
        return _svd(w, full_matrices=full_matrices)
    original_device = w.device
    w = w.to(accelerator)
    u, s, v = _svd(w)
    return u, s, v


@torch.no_grad()
def upscale_to_smile_linear(
    base: nn.Linear, experts: List[nn.Linear], target: SmileLinear, accelerator=None
):
    """
    Upscale a base linear layer to a SmileLinear layer using expert models.

    Args:
        base (nn.Linear): The base linear layer.
        experts (List[nn.Linear]): A list of expert linear layers.
        target (SmileLinear): The target SmileLinear layer.
        accelerator (optional): The device to perform the computation on. Defaults to None.

    Returns:
        SmileLinear: The upscaled SmileLinear layer.
    """
    w = base.weight
    w_ft_list = [e.weight for e in experts]
    dw_list = [w_ft - w for w_ft in w_ft_list]

    if _is_all_zeros(dw_list):
        raise ExpertNotTrainedError("Expert models are not trained")

    rank_of_router = target.rank_of_router
    rank_of_expert = target.rank_of_expert
    num_local_experts = target.num_local_experts
    svd_list = [svd(dw, accelerator=accelerator) for dw in dw_list]

    # gate
    gate_weight = []
    for u, s, v in svd_list:
        gate_weight.append(v[:, :rank_of_router].T)
    gate_weight = (
        torch.stack(gate_weight, dim=0)
        .reshape(num_local_experts * rank_of_router, -1)
        .contiguous()
    )

    target.gate.load_state_dict({"weight": gate_weight})

    # shared linear
    target.shared_linear.load_state_dict(base.state_dict())

    # experts
    if rank_of_expert > 0:
        for expert_idx, target_expert in enumerate(target.experts):
            u, s, v = svd_list[expert_idx]
            u = u[:, :rank_of_expert]
            s = s[:rank_of_expert]
            v = v[:, :rank_of_expert]
            state_dict = {"u": u, "svh": (s * v).T}
            if experts[expert_idx].bias is not None:
                state_dict["bias"] = experts[expert_idx].bias.data
            target_expert.load_state_dict(state_dict)
    else:
        for expert_idx, target_expert in enumerate(target.experts):
            target_expert.load_state_dict(
                state_dict_sub(experts[expert_idx].state_dict(), base.state_dict())
            )

    return target


class SmileMistralUpscalingAlgorithm(ModelFusionAlgorithm, SimpleProfilerMixin):
    R"""
    SmileMistralUpscalingAlgorithm is a model fusion algorithm designed to upscale
    a pretrained Mistral model using a set of fine-tuned expert models. The algorithm
    leverages Singular Value Decomposition (SVD) to merge the weights of the pretrained
    model and the expert models into a new upscaled model.

    Attributes:
        modelpool (BaseModelPool): The pool of models to be used for upscaling.
        config (dict): Configuration parameters for the upscaling process.

    Methods:
        run(modelpool: BaseModelPool) -> SmileMistralForCausalLM:
            Executes the upscaling process and returns the upscaled model.

        merge(pretrained_model: MistralForCausalLM, finetuned_models: List[MistralForCausalLM]) -> SmileMistralForCausalLM:
            Merges the pretrained model with the fine-tuned models to create an upscaled model.
    """

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool) -> SmileMistralForCausalLM:
        """
        Executes the upscaling process.

        Args:
            modelpool (ModelPool): The pool of models to be used for upscaling.

        Returns:
            SmileMistralForCausalLM: The upscaled model.
        """
        self.modelpool = modelpool = to_modelpool(modelpool)
        config = self.config

        print(config)
        if config.model_path is not None and os.path.exists(config.model_path):
            log.info(f"Loading model from {config.model_path}")
            model = torch.load(config.model_path)
            print_parameters(model)
            return model

        with self.profile("load pretrained model"):
            pretrained_model = modelpool.load_model("_pretrained_")
        with self.profile("load fine-tuned model"):
            finetuned_models = [
                m for m in tqdm(modelpool.models(), total=len(modelpool.model_names))
            ]

        if config.device == "cuda" and torch.cuda.is_available():
            pretrained_model = pretrained_model.cuda()
            finetuned_models = [m.cuda() for m in finetuned_models]

        with self.profile("merge model"):
            model = self.merge(pretrained_model, finetuned_models)

        self.print_profile_summary()
        print_parameters(model)
        print(model)

        if config.model_dtype is not None:
            model.to(dtype=parse_dtype(config.model_dtype))

        if config.model_path is not None:
            if os.path.dirname(config.model_path):
                os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
            log.info(f"Saving model to {config.model_path}")
            pretrained_model_config = self.modelpool.get_model_config("_pretrained_")
            pretrained_path = pretrained_model_config.get(
                "path", pretrained_model_config["pretrained_model_name_or_path"]
            )
            tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
            tokenizer.save_pretrained(config.model_path)
            model.save_pretrained(config.model_path)

        return model

    def merge(
        self,
        pretrained_model: MistralForCausalLM,
        finetuned_models: List[MistralForCausalLM],
    ):
        """
        Merges the pretrained model with the fine-tuned models to create an upscaled model.

        Args:
            pretrained_model (MistralForCausalLM): The pretrained model.
            finetuned_models (List[MistralForCausalLM]): A list of fine-tuned models.

        Returns:
            SmileMistralForCausalLM: The upscaled model.
        """
        config = self.config

        with init_empty_weights():
            pretrained_model_config = self.modelpool.get_model_config("_pretrained_")
            pretrained_path = pretrained_model_config.get(
                "path", pretrained_model_config["pretrained_model_name_or_path"]
            )
            base_config = AutoConfig.from_pretrained(pretrained_path)
            model_config = SmileMistralConfig(
                num_experts_per_tok=config.num_experts_per_tok,
                rank_of_router=config.rank_of_router,
                rank_of_expert=config.rank_of_expert,
                num_local_experts=len(finetuned_models),
                **base_config.to_dict(),
            )
            model = SmileMistralForCausalLM(model_config)

        model.to(dtype=pretrained_model.dtype).to_empty(device="cpu")

        # copy pretrained model weights
        state_dict = model.state_dict()
        pretrained_state_dict = dict(pretrained_model.state_dict())
        for key in list(pretrained_state_dict.keys()):
            if key not in state_dict:
                pretrained_state_dict.pop(key)
        model.load_state_dict(pretrained_state_dict, strict=False)

        # upscale model
        for layer_idx in tqdm(
            range(len(pretrained_model.model.layers)),
            "Upscaling Modules (layer)",
            dynamic_ncols=True,
        ):
            pretrained_layer: MistralDecoderLayer = pretrained_model.model.layers[
                layer_idx
            ]
            finetuned_layers: List[MistralDecoderLayer] = [
                m.model.layers[layer_idx] for m in finetuned_models
            ]

            target_layer: SmileMistralDecoderLayer = model.model.layers[layer_idx]

            for n in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                try:
                    upscale_to_smile_linear(
                        base=getattr(pretrained_layer.self_attn, n),
                        experts=[getattr(m.self_attn, n) for m in finetuned_layers],
                        target=getattr(target_layer.self_attn, n),
                        accelerator=config.accelerator,
                    )
                except ExpertNotTrainedError:
                    setattr(
                        target_layer.self_attn,
                        n,
                        getattr(pretrained_layer.self_attn, n),
                    )

            for n in ["gate_proj", "up_proj", "down_proj"]:
                try:
                    upscale_to_smile_linear(
                        base=getattr(pretrained_layer.mlp, n),
                        experts=[getattr(m.mlp, n) for m in finetuned_layers],
                        target=getattr(target_layer.mlp, n),
                        accelerator=config.accelerator,
                    )
                except ExpertNotTrainedError:
                    setattr(
                        target_layer.mlp,
                        n,
                        getattr(pretrained_layer.mlp, n),
                    )

        return model
