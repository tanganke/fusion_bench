import logging
import os
from copy import deepcopy
from typing import Dict, List, Tuple  # noqa: F401

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor, nn
from tqdm.auto import tqdm

from fusion_bench.method import BaseAlgorithm
from fusion_bench.method.simple_average import simple_average
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.models.smile_moe.linear_from_module import (
    ExpertNotTrainedError,
    SmileCompressedLinear,
    SmileGate,
    SmileMoELinear,
)
from fusion_bench.models.utils import get_attr, set_attr
from fusion_bench.utils.parameters import print_parameters

log = logging.getLogger(__name__)


class SmileUpscalingAlgorithm(
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    _linear_layer_cls = (nn.Linear,)
    _config_mapping = BaseAlgorithm._config_mapping | {
        "device": "device",
        "upscaling_accelerator": "upscaling_accelerator",
        "full_matrices": "full_matrices",
        "gate_k": "gate_k",
        "k": "k",
        "top_k": "top_k",
        "routing_use_diff": "routing_use_diff",
        "average_experts": "average_experts",
        "model_path": "model_path",
    }

    def __init__(
        self,
        *,
        device: str = "cuda",
        upscaling_accelerator: str = None,
        full_matrices: bool = True,
        gate_k: int = 256,
        k: int = 256,
        top_k: int = 1,
        routing_use_diff: bool = True,
        average_experts: bool = False,
        model_path: str = None,
        **kwargs,
    ):
        """
        Initialize the SmileUpscalingAlgorithm.

        Args:
            device (str): The device to perform the computation on.
            upscaling_accelerator (str): The device to perform the SVD computation on.
            full_matrices (bool): Whether to compute the full-sized U and V matrices.
            gate_k (int): The number of singular values to keep for the gate.
            k (int): The number of singular values to keep for the experts.
            top_k (int): The number of top experts to select.
            routing_use_diff (bool): Whether to use weight differences for routing.
            average_experts (bool): Whether to average the experts.
            model_path (str): The path to save/load the model.
            **kwargs: Additional arguments.
        """
        super().__init__()
        self.device = device
        self.upscaling_accelerator = upscaling_accelerator
        self.full_matrices = full_matrices
        self.gate_k = gate_k
        self.k = k
        self.top_k = top_k
        self.routing_use_diff = routing_use_diff
        self.average_experts = average_experts
        self.model_path = model_path
        for key, value in kwargs.items():
            log.warning(f"Unrecognized argument: {key}")
            setattr(self, key, value)

        # print `self.config` as yaml
        print(f"=== Config for `{type(self).__name__}` ===")
        print(OmegaConf.to_yaml(self.config))
        print(f"=== Config for `{type(self).__name__}` ===")

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        """
        Executes the upscaling process.

        Args:
            modelpool (ModelPool): The pool of models to be used for upscaling.

        Returns:
            nn.Module: The upscaled model.
        """
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        if self.config.model_path is not None and os.path.exists(
            self.config.model_path
        ):
            log.info(f"Loading model from {self.config.model_path}")
            model = torch.load(self.config.model_path)
            print_parameters(model)
            return model

        with self.profile("loading model"):
            # load models and move to GPU if available
            with self.profile("load pretrained model"):
                pretrained_model = modelpool.load_model("_pretrained_")
            with self.profile("load fine-tuned model"):
                finetuned_models = [
                    m
                    for m in tqdm(modelpool.models(), total=len(modelpool.model_names))
                ]

            if self.config.device == "cuda" and torch.cuda.is_available():
                pretrained_model = pretrained_model.cuda()
                finetuned_models = [m.cuda() for m in finetuned_models]

        with self.profile("merge model"):
            model = self.merge(pretrained_model, finetuned_models)

        self.print_profile_summary()
        if self.config.model_path is not None:
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            log.info(f"Saving model to {self.config.model_path}")
            torch.save(model, self.config.model_path)
        print_parameters(model)
        return model

    def merge(
        self,
        pretrained_model: nn.Module,
        finetuned_models: List[nn.Module],
        in_place: bool = True,
    ):
        """
        Merges the pretrained model with the fine-tuned models to create an upscaled model.

        Args:
            pretrained_model (nn.Module): The pretrained model.
            finetuned_models (List[nn.Module]): A list of fine-tuned models.
            in_place (bool): If True, modifies the pretrained model in place. Otherwise, creates a copy.

        Returns:
            nn.Module: The merged model.
        """
        if in_place:
            model = pretrained_model
        else:
            model = deepcopy(pretrained_model)

        self._upscale_submodules(model, finetuned_models)
        return model

    def _upscale_linear_layer(
        self,
        pretrained_model,
        finetuned_models,
        name: str,
    ):
        """
        Upscale a linear layer by merging it with the corresponding layers from the fine-tuned models.

        Args:
            pretrained_model (nn.Module): The pretrained model.
            finetuned_models (List[nn.Module]): A list of fine-tuned models.
            name (str): The name of the linear layer to upscale.
        """
        config = self.config

        name_list = name.split(".")
        module = get_attr(pretrained_model, name_list)
        experts = [get_attr(m, name_list) for m in finetuned_models]
        try:
            moe_linear = SmileMoELinear(
                module,
                experts,
                gate_k=config.gate_k,
                k=config.k,
                top_k=config.top_k,
                routing_use_diff=self.routing_use_diff,
                full_matrices=self.full_matrices,
                upscaling_accelerator=self.upscaling_accelerator,
            )
        except ExpertNotTrainedError:
            print(f"skip {name} because the experts are not trained.")
            return
        set_attr(pretrained_model, name_list, moe_linear)
        # remove the original module from fine-tuned models to save memory
        for m in finetuned_models:
            set_attr(m, name_list, None)

    def _average_experts(self, pretarined_model, finetuned_models, name: str):
        """
        Average the experts for a given layer.

        Args:
            pretarined_model (nn.Module): The pretrained model.
            finetuned_models (List[nn.Module]): A list of fine-tuned models.
            name (str): The name of the layer to average.
        """
        name_list = name.split(".")
        experts = [get_attr(m, name_list) for m in finetuned_models]
        averaged_module = simple_average(experts)
        set_attr(pretarined_model, name_list, averaged_module)

    def _upscale_submodules(
        self,
        pretrained_model: nn.Module,
        finetuned_models: List[nn.Module],
        tqdm_desc: str = "Upscaling Linear Modules",
    ):
        """
        Upscales the submodules of the pretrained model by merging them with the corresponding submodules from the fine-tuned models.

        Args:
            pretrained_model (nn.Module): The pretrained model.
            finetuned_models (List[nn.Module]): A list of fine-tuned models.
            tqdm_desc (str): Description for the tqdm progress bar.
        """
        config = self.config
        for name, module in tqdm(
            tuple(pretrained_model.named_modules()),
            tqdm_desc,
            leave=False,
            dynamic_ncols=True,
        ):
            if isinstance(module, self._linear_layer_cls):
                self._upscale_linear_layer(
                    pretrained_model=pretrained_model,
                    finetuned_models=finetuned_models,
                    name=name,
                )
            elif config.average_experts and len(tuple(module.named_modules())) == 1:
                # if the module is a leaf module, we perform a parameter average
                self._average_experts(pretrained_model, finetuned_models, name)
