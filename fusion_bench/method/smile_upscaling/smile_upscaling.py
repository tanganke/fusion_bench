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
from fusion_bench.models.utils import get_attr, set_attr
from fusion_bench.utils.parameters import print_parameters

log = logging.getLogger(__name__)


class ExpertNotTrainedError(Exception):
    pass


def _is_all_zeros(tensor: Tensor | List[Tensor]) -> bool:
    """
    Check if a tensor or a list of tensors are all zeros.

    Args:
        tensor (Tensor | List[Tensor]): A tensor or a list of tensors.

    Returns:
        bool: True if all elements are zeros, False otherwise.
    """
    if isinstance(tensor, Tensor):
        return torch.allclose(tensor, torch.zeros_like(tensor))
    else:
        return all(_is_all_zeros(t) for t in tensor)


def _svd(w: Tensor, full_matrices=True) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform Singular Value Decomposition (SVD) on a tensor.

    Args:
        w (Tensor): The input tensor.
        full_matrices (bool): Whether to compute the full-sized U and V matrices.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The U, S, and V matrices from SVD.
    """
    u, s, vh = torch.linalg.svd(
        w, full_matrices=full_matrices, driver="gesvd" if w.is_cuda else None
    )
    v = vh.T
    return u, s, v


def svd(
    w: Tensor, full_matrices=True, accelerator=None
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform SVD on a tensor, optionally using a specified accelerator.

    Args:
        w (Tensor): The input tensor.
        full_matrices (bool): Whether to compute the full-sized U and V matrices.
        accelerator (str): The device to perform the computation on.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The U, S, and V matrices from SVD.
    """
    if accelerator is None:
        return _svd(w, full_matrices=full_matrices)
    original_device = w.device
    w = w.to(accelerator)
    u, s, v = _svd(w)
    return u.to(original_device), s.to(original_device), v.to(original_device)


class SmileGate(nn.Module):
    def __init__(
        self,
        input_features: int,
        w_diff_list: List[Tensor],
        k: int,
        svd_list=None,  # cached `svd_list`, pass it to avoid recomputing
        upscaling_accelerator=None,
    ):
        """
        Initialize the SmileGate module.

        Args:
            input_features (int): The number of input features.
            w_diff_list (List[Tensor]): A list of weight difference tensors.
            k (int): The number of singular values to keep.
            svd_list (List[Tuple[Tensor, Tensor, Tensor]]): Cached SVD results.
            upscaling_accelerator (str): The device to perform the computation on.
        """
        super().__init__()
        self.input_features = input_features
        self.num_experts = len(w_diff_list)
        weights = []
        for i, w_diff in enumerate(w_diff_list):
            if svd_list is None:
                u, s, v = svd(w_diff, accelerator=upscaling_accelerator)
            else:
                u, s, v = svd_list[i]
            u = u[:, :k]
            s = s[:k]
            v = v[:, :k]

            # weights.append((s * v).T)
            weights.append(v.T)
        self.k = s.size(0)  # k is the actual k after truncation

        weights = (
            torch.stack(weights, dim=0)
            .reshape(self.num_experts * self.k, -1)
            .contiguous()
        )
        self.weights = nn.Parameter(
            weights
        )  # weights should be a tensor of shape (num_experts * k, n)

    def forward(self, x: Tensor):
        """
        Forward pass of the SmileGate module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The routing weights.
        """
        batch_size = x.size(0)
        if self.num_experts == 1:
            return torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)

        routing_weights = F.linear(x, self.weights).view(
            batch_size, self.num_experts, self.k
        )
        routing_weights = routing_weights.norm(p=2, dim=2)
        return routing_weights


class SmileCompressedLinear(nn.Module):
    def __init__(self, model: nn.Linear, k: int, svd_cache=None):
        """
        Initialize the SmileCompressedLinear module.

        Args:
            model (nn.Linear): The linear model to compress.
            k (int): The number of singular values to keep.
            svd_cache (Tuple[Tensor, Tensor, Tensor]): Cached SVD results.
        """
        super().__init__()
        if svd_cache is None:
            u, s, v = svd(model.weight)
        else:
            u, s, v = svd_cache
        if k > 0:
            u = u[:, :k]
            s = s[:k]
            v = v[:, :k]

        self.u = nn.Parameter(u)
        self.svh = nn.Parameter((s * v).T)

        if model.bias is not None:
            self.bias = nn.Parameter(model.bias.data, requires_grad=True)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        Forward pass of the SmileCompressedLinear module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x = F.linear(x, self.svh)
        x = F.linear(x, self.u, self.bias)
        return x


class SmileMoELinear(nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        pretrained_model: nn.Linear,
        finetuned_models: List[nn.Linear],
        gate_k: int,
        k: int,
        top_k: int = 1,
        full_matrices=True,
        upscaling_accelerator=None,
        routing_use_diff=True,
    ):
        """
        Initialize the SmileMoELinear module.

        Args:
            pretrained_model (nn.Linear): The pretrained linear model.
            finetuned_models (List[nn.Linear]): A list of fine-tuned linear models.
            gate_k (int): The number of singular values to keep for the gate.
            k (int): The number of singular values to keep for the experts.
            top_k (int): The number of top experts to select.
            full_matrices (bool): Whether to compute the full-sized U and V matrices.
            upscaling_accelerator (str): The device to perform the computation on.
            routing_use_diff (bool): Whether to use weight differences for routing.
        """
        super().__init__()
        self.num_experts = len(finetuned_models)
        self.top_k = top_k
        self.k = k
        self.gate_k = gate_k
        self.in_features = pretrained_model.in_features
        self.out_features = pretrained_model.out_features

        w_diff_list = [m.weight - pretrained_model.weight for m in finetuned_models]
        if _is_all_zeros(w_diff_list):
            # All fine-tuned models are identical to the pretrained model
            raise ExpertNotTrainedError()

        if routing_use_diff or k > 0:
            svd_cache_list = [
                svd(w, full_matrices=full_matrices, accelerator=upscaling_accelerator)
                for w in w_diff_list
            ]  # the svd cache list to avoid recomputing

        # construct the gate network
        if routing_use_diff:
            self.gate = SmileGate(
                input_features=self.in_features,
                w_diff_list=w_diff_list,
                k=gate_k,
                svd_list=svd_cache_list,
                upscaling_accelerator=upscaling_accelerator,
            )
        else:
            self.gate = SmileGate(
                input_features=self.in_features,
                w_diff_list=[m.weight for m in finetuned_models],
                k=gate_k,
                svd_list=None,
                upscaling_accelerator=upscaling_accelerator,
            )

        # construct experts
        for m, w_diff in zip(finetuned_models, w_diff_list):
            m.weight.data = w_diff
        if k > 0:
            experts = [
                SmileCompressedLinear(m, k, svd_cache=svd_cache)
                for m, svd_cache in zip(finetuned_models, svd_cache_list)
            ]
        else:
            # if k is not set (<0), we use the full fine-tuned model
            experts = finetuned_models
        self.experts = nn.ModuleList(experts)

        if pretrained_model.bias is not None:
            for m in experts:
                m.bias.data = m.bias.data - pretrained_model.bias
        # assign the pretrained model (the shared part)
        self.pretrained_model = pretrained_model

    def forward(self, hidden_states: Tensor):
        """
        Forward pass of the SmileMoELinear module.

        Args:
            hidden_states (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        pretrained_out = self.pretrained_model(hidden_states)

        input_shape = hidden_states.size()
        hidden_states = hidden_states.view(-1, self.in_features)

        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1)
        # sample the expert according to the routing weights
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros(
            (hidden_states.size(0), self.out_features),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, self.in_features)
            if current_state.numel() == 0:
                continue
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(
            *input_shape[:-1], self.out_features
        )
        final_hidden_states = pretrained_out + final_hidden_states
        return final_hidden_states

    @property
    def weight(self):
        """
        Mimic linear layer. Bacause in some cases, user might indicate the device (or dtype of parameters) of the linear layer using `linear_layer.weight.device`
        """
        return self.pretrained_model.weight

    @property
    def bias(self):
        return self.pretrained_model.bias

    def __repr__(self):
        return (
            f"SingularMoELinear("
            f"in_features={self.pretrained_model.in_features}, "
            f"out_features={self.pretrained_model.out_features}, "
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"gate_k={self.gate_k}, "
            f"k={self.k}"
            f")"
        )


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

        with self.profile("load pretrained model"):
            pretrained_model = modelpool.load_model("_pretrained_")
        with self.profile("load fine-tuned model"):
            finetuned_models = [
                m for m in tqdm(modelpool.models(), total=len(modelpool.model_names))
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
