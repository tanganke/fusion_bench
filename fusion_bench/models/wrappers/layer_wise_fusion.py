import functools
import logging
from copy import deepcopy
from typing import (  # noqa: F401
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    TypeVar,
)

import torch
from torch import Tensor, nn
from torch.func import functional_call

from fusion_bench.models.utils import del_attr, get_attr, set_attr
from fusion_bench.utils.type import StateDictType, TorchModelType

__all__ = ["get_layer_wise_weights", "fuse_weights", "LayerWiseMergedModel"]

log = logging.getLogger(__name__)


def get_layer_wise_weights(
    num_models: int,
    num_layers: int,
    init_values: float = None,
    dtype: torch.dtype = torch.float32,
):
    """
    Return a tensor of layer-wise weights for the given number of models and layers.

    Args:
        num_models (int): The number of models to fuse.
        num_layers (int): The number of layers in each model.
        init_values (float, optional): The initial value for each weight. Defaults to 1.0 / num_models.
        dtype (torch.dtype): dtype of weights. This should be the same with model dtype.

    Returns:
        Tensor: A tensor of shape (num_models, num_layers) containing the layer-wise weights.
    """
    assert num_models >= 1, f"num_models must be >= 1, got {num_models}"
    assert num_layers >= 1, f"num_layers must be >= 1, got {num_layers}"
    if init_values is None:
        init_values = 1.0 / num_models
    return torch.full((num_models, num_layers), init_values, dtype=dtype)


def _fuse_weights(layer_wise_weight: Tensor, tensors: List[Tensor]):
    """
    Fuse the layer-wise weights with the given state dictionaries.

    Args:
        layer_wise_weight (Tensor): A tensor of shape (num_models,) containing the layer-wise weights.
        state_dicts (List[Tensor]): A list of state dictionaries, each containing the weights for a single layer.

    Returns:
        Tensor: A tensor of shape (num_params,) containing the fused weights.
    """
    assert len(layer_wise_weight) == len(
        tensors
    ), f"layer_wise_weight.shape={layer_wise_weight.shape}, len(tensors)={len(tensors)}"
    return sum(
        layer_wise_weight[i] * w.to(layer_wise_weight.device)
        for i, w in enumerate(tensors)
    )


def fuse_weights(
    layer_wise_weight: Tensor, state_dicts: List[StateDictType]
) -> StateDictType:
    """
    Fuse the weights of multiple models using layer-wise fusion.

    Args:
        layer_wise_weight (Tensor): A tensor of shape (num_models, num_layers) representing the weight of each layer for each model.
        state_dicts (List[StateDict]): A list of state dictionaries, one for each model.

    Returns:
        A dictionary mapping each weight tensor key to the fused weight tensor.
    """
    num_models = len(state_dicts)
    num_layers = len(state_dicts[0])
    assert layer_wise_weight.shape == (
        num_models,
        num_layers,
    ), f"layer_wise_weight.shape={layer_wise_weight.shape}, expected (num_models, num_layers): ({num_models}, {num_layers})"
    return {
        k: _fuse_weights(
            layer_wise_weight[:, i], [state_dict[k] for state_dict in state_dicts]
        )
        for i, k in enumerate(state_dicts[0].keys())
    }


class LayerWiseMergedModel(nn.Module, Generic[TorchModelType]):
    _merged_state_dict: StateDictType = None

    def __init__(
        self,
        layer_wise_weight: Tensor,
        pretrained_model: TorchModelType,
        finetuned_models: List[TorchModelType],
        clamp_weights: bool = True,
        tie_weights: bool = False,
        strict: bool = True,
        sparsity_ratio: Optional[float] = None,
        normalized_merging_weights: bool = False,
    ):
        R"""
        This class wraps a pretrained model and a list of finetuned models, and merges the weights of the finetuned models into the pretrained model using layer-wise fusion.

        Reference:

            (ICLR 2024) Yang E, Wang Z, Shen L, et al. Adamerging: Adaptive model merging for multi-task learning. https://arxiv.org/pdf/2310.02575

        Args:
            layer_wise_weight (Tensor): A tensor of shape (num_models, num_layers) representing the weight of each layer for each model.
            pretrained_model (nn.Module): The pretrained model to merge the weights into.
            finetuned_models (List[nn.Module]): A list of finetuned models to merge the weights from. This should have the same architecture as the pretrained model. We use these models to compute the task vectors.
            clamp_weights (bool, optional): If True, the layer-wise weights will be clamped to [0, 1]. Defaults to True.
            tie_weights (bool, optional): This option passes the `tie_weights` argument to the `functional_call` function. Defaults to False.
            strict (bool, optional): This option passes the `strict` argument to the `functional_call` function. Defaults to True.
            sparsity_ratio (float, optional): If `sparsity_ratio` is provided, the task vector will be pruned before merging. A high spasity level can save the memory usage during merging.
            normalized_merging_weights (bool, optional): If True, the layer-wise weights will be normalized for each layer, so that the sum of weights across models for each layer is 1. Defaults to False.
        """
        super().__init__()
        self.clamp_weights = clamp_weights
        self.tie_weights = tie_weights
        self.strict = strict
        self.sparsity_ratio = sparsity_ratio
        self.nromalized_merging_weights = normalized_merging_weights

        self.merge_weight = nn.Parameter(layer_wise_weight, requires_grad=True)

        for name, param in pretrained_model.named_parameters():
            if not param.requires_grad:
                for m in finetuned_models:
                    del_attr(m, name.split("."))
            else:
                for m in finetuned_models:
                    get_attr(m, name.split(".")).data = (
                        get_attr(m, name.split(".")) - param
                    )

        self.pretrained_model = pretrained_model.requires_grad_(False)
        for m in finetuned_models:
            m.requires_grad_(False)

        self.task_vectors = nn.ModuleList(finetuned_models)

        # if `sparisty_ratio` is given, pruning the task vectors.
        if sparsity_ratio is not None:
            from fusion_bench.method.pruning.prune_utils import (
                unstructured_magnitude_prune_,
            )

            for name, param in self.task_vectors.named_parameters():
                if param.dim() != 2:
                    continue
                print(f"pruning {name}")
                pruned_param = unstructured_magnitude_prune_(
                    param.data.clone(), torch.abs, sparsity_ratio=sparsity_ratio
                )
                set_attr(
                    self.task_vectors,
                    name.split("."),
                    nn.Parameter(pruned_param.to_sparse(), requires_grad=False),
                )

    @property
    def forward_model(self):
        return functools.partial(
            functional_call,
            self.pretrained_model,
            self._merged_state_dict,
            tie_weights=self.tie_weights,
            strict=self.strict,
        )

    def merge_and_unload(self, task_vector_mask: Optional[Dict[str, Tensor]] = None):
        self.merge_weights(task_vector_mask=task_vector_mask)
        self.pretrained_model.load_state_dict(self._merged_state_dict)
        return self.pretrained_model

    def merge_weights(self, task_vector_mask: Optional[Dict[str, Tensor]] = None):
        """
        Merges the weights of the model.
        Call this after each update step.
        """
        if self.clamp_weights:
            layer_wise_weight = self.merge_weight.clamp(0, 1)
        else:
            layer_wise_weight = self.merge_weight
        if self.nromalized_merging_weights:
            # normalize the weights for each layer, so that the sum of weights across models for each layer is 1.
            layer_wise_weight = layer_wise_weight.softmax(dim=0)

        state_dict = self.pretrained_model.state_dict(keep_vars=True)
        # shape of layer_wise_weight: (num_models, num_layers)
        for weight, task_vector in zip(layer_wise_weight, self.task_vectors):
            assert len(list(task_vector.named_parameters())) == weight.size(0)
            if task_vector_mask is not None:
                weight = [
                    w * task_vector_mask[name]
                    for w, (name, param) in zip(weight, task_vector.named_parameters())
                ]
            for w, (name, param) in zip(weight, task_vector.named_parameters()):
                state_dict[name] = state_dict[name] + param * w
        self._merged_state_dict = state_dict

        return state_dict

    def forward(self, *args, **kwargs):
        if self._merged_state_dict is None:
            self.merge_weights()
        return self.forward_model(args=args, kwargs=kwargs)

    # def __getattr__(self, name: str) -> Any:
    #     try:
    #         return super().__getattr__(name)
    #     except AttributeError:
    #         attr = getattr(self.model, name)
    #         if isinstance(attr, Callable):
    #             warnings.warn(
    #                 f"forwarding `{name}` to the underlying model", UserWarning
    #             )
    #         return attr

    # def __setattr__(self, name: str, value: Any) -> None:
    #     try:
    #         super().__setattr__(name, value)
    #     except AttributeError:
    #         setattr(self.model, name, value)


def merge_weights(module: nn.Module):
    """
    Merges the weights for all `LayerWiseMergedModel` instances within the given module.

    Args:
        module (nn.Module): The module to process.
    """
    if isinstance(module, LayerWiseMergedModel):
        module.merge_weights()
        return
    else:
        for submodule in module.children():
            merge_weights(submodule)


def merge_and_unload(module: nn.Module):
    """
    Merges and unloads all `LayerWiseMergedModel` instances within the given module.

    Args:
        module (nn.Module): The module to process.

    Returns:
        nn.Module: The updated module with merged weights.
    """
    if isinstance(module, LayerWiseMergedModel):
        return module.merge_and_unload()
    else:
        for name, submodule in module.named_children():
            need_merge = isinstance(submodule, LayerWiseMergedModel)
            submodule = merge_and_unload(submodule)
            if need_merge:
                setattr(module, name, submodule)
        return module


def fix_other_parts(module: nn.Module):
    """
    Sets all parameters in the module to not require gradients, except for the merge weights
    in `LayerWiseMergedModel` instances.

    Args:
        module (nn.Module): The module to process.

    Returns:
        nn.Module: The module with updated parameter requirements.
    """
    module.requires_grad_(False)
    for submodule in module.modules():
        if isinstance(submodule, LayerWiseMergedModel):
            submodule.merge_weight.requires_grad_(True)
    return module
