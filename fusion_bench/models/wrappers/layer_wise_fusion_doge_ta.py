import copy
import functools
import logging
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, Iterator, List, Optional  # noqa: F401

import lightning as L
import torch
from torch import Tensor, nn
from torch.func import functional_call

from fusion_bench.models.utils import del_attr, get_attr, set_attr
from fusion_bench.utils.state_dict_arithmetic import state_dict_add
from fusion_bench.utils.type import StateDictType

from .layer_wise_fusion import fuse_weights, get_layer_wise_weights

__all__ = ["get_layer_wise_weights", "fuse_weights", "LayerWiseMergedModel"]

log = logging.getLogger(__name__)


class LayerWiseMergedModel(nn.Module):
    _merged_state_dict: StateDictType = None

    def __init__(
        self,
        layer_wise_weight: Tensor,
        pretrained_model: nn.Module,
        finetuned_models: List[nn.Module],
        clamp_weights: bool = True,
        tie_weights: bool = False,
        strict: bool = True,
        sparsity_ratio: Optional[float] = None,
        normalized_merging_weights: bool = False,
    ):
        R"""
        This class wraps a pretrained model and a list of finetuned models, and merges the weights of the finetuned models into the pretrained model using layer-wise fusion.

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
        if torch.cuda.is_available():
            self._fabric = L.Fabric(devices=1)
            self._fabric.launch()
        self.clamp_weights = clamp_weights
        self.tie_weights = tie_weights
        self.strict = strict
        self.sparsity_ratio = sparsity_ratio
        self.nromalized_merging_weights = normalized_merging_weights

        pretrained_sd = pretrained_model.state_dict(keep_vars=True)
        filtered_keys = [
            k
            for k in pretrained_sd.keys()
            if ("encoder" in k and "layer_norm" not in k and "weight" in k)
        ]
        self.merge_weight = nn.Parameter(
            layer_wise_weight[:, : len(filtered_keys)], requires_grad=True
        )
        task_vectors = []
        for m in finetuned_models:
            m.requires_grad_(False)
        self.pretrained_model = pretrained_model.requires_grad_(False)
        for model in finetuned_models:
            model_sd = model.state_dict(keep_vars=True)
            filtered_task_vector = {
                k: (model_sd[k] - pretrained_sd[k]) for k in filtered_keys
            }
            if self._fabric is not None:
                filtered_task_vector = self._fabric.to_device(filtered_task_vector)
            task_vectors.append(filtered_task_vector)

        self.projection = {}
        for layer_name in task_vectors[0].keys():
            for i, vector in enumerate(task_vectors):
                layer_vector = vector[layer_name]
                u, s, v = torch.linalg.svd(layer_vector, full_matrices=False)
                if i == 0:
                    print(f"Computed SVD for {layer_name}...")
                    sum_u = torch.zeros_like(u, device=layer_vector.device)
                    sum_s = torch.zeros_like(s, device=layer_vector.device)
                    sum_v = torch.zeros_like(v, device=layer_vector.device)

                reduced_index_s = int(s.shape[0] / len(task_vectors))

                # select only the first reduced_index_s columns of u and place them
                sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                    :, :reduced_index_s
                ]
                sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                    :reduced_index_s
                ]
                # select only the first reduced_index_s rows of v and place them
                sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                    :reduced_index_s, :
                ]
            u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
            # u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
            layer_proj = torch.matmul(
                u_u[:, : int(s.shape[0] / len(task_vectors))],
                u_u[:, : int(s.shape[0] / len(task_vectors))].T,
            )
            self.projection[layer_name] = layer_proj

        self.delta = [
            {
                k: torch.zeros_like(v).clone().requires_grad_()
                for k, v in task_vector.items()
            }
            for task_vector in task_vectors
        ]
        if self._fabric is not None:
            self.delta = self._fabric.to_device(self.delta)
        self.lamdas = self.compute_layer_lamdas(task_vectors)

        for layer_name in task_vectors[0].keys():
            optimizer = torch.optim.Adam(
                [delta[layer_name] for delta in self.delta], lr=1e-4
            )
            layer_vectors = torch.stack([vec[layer_name] for vec in task_vectors])
            layer_lamdas = torch.stack([lamdas[layer_name] for lamdas in self.lamdas])
            for _ in range(400):
                optimizer.zero_grad()
                layer_delta = torch.stack([de[layer_name] for de in self.delta])
                loss = self.taskvector_loss(layer_vectors, layer_delta, layer_lamdas)
                print(f"Epoch: {_}, Layer: {layer_name}, Loss: {loss.item()}")
                self._fabric.backward(loss)
                for delta in self.delta:
                    grad_proj = (
                        self.projection[layer_name] @ delta[layer_name].grad.detach()
                    )
                    delta[layer_name].grad.data = delta[layer_name].grad.data.sub_(
                        grad_proj
                    )
                optimizer.step()
                for delta in self.delta:
                    for param in delta.values():
                        param.grad = None
        del self.projection
        self.delta = [
            {key: param.detach().cpu() for key, param in delta.items()}
            for delta in self.delta
        ]
        self.lamdas = [
            {key: param.cpu() for key, param in lamdas.items()}
            for lamdas in self.lamdas
        ]
        task_vectors = [
            {k: v.cpu() for k, v in task_vector.items()} for task_vector in task_vectors
        ]
        flat_vectors = []
        vector_masks = []
        for idx, task_vector in enumerate(task_vectors):
            flat_vector = self.state_dict_to_vector(task_vector)
            vector_mask = self.topk_values_mask(flat_vector, K=30)
            flat_vectors.append(flat_vector)
            vector_masks.append(vector_mask)
        flat_deltas = [self.state_dict_to_vector(delta) for delta in self.delta]
        self.task_vectors = [
            self.vector_to_state_dict(
                (flat_vector + flat_delta) * vector_mask, self.delta[0]
            )
            for flat_vector, flat_delta, vector_mask in zip(
                flat_vectors, flat_deltas, vector_masks
            )
        ]
        if self._fabric is not None:
            self.task_vectors = self._fabric.to_device(self.task_vectors)

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

    def topk_values_mask(self, M, K):
        if K > 1:
            K /= 100

        original_shape = M.shape
        if M.dim() == 1:
            M = M.unsqueeze(0)

        n, d = M.shape
        k = int(d * K)
        k = d - k  # Keep top k elements instead of bottom k elements

        # Find the k-th smallest element by magnitude for each row
        kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
        # Create a mask tensor with True for the top k elements in each row
        mask = M.abs() >= kth_values
        final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

        return final_mask

    def state_dict_to_vector(self, state_dict, remove_keys=[]):
        """
        Convert a state dictionary to a vector, removing specified keys.

        Args:
            state_dict (dict): The state dictionary to convert.
            remove_keys (list): List of keys to remove from the state dictionary.

        Returns:
            Tensor: A vector representation of the state dictionary.
        """
        shared_state_dict = copy.deepcopy(state_dict)
        for key in remove_keys:
            if key in shared_state_dict:
                del shared_state_dict[key]
        sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
        return nn.utils.parameters_to_vector(
            [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
        )

    def vector_to_state_dict(self, vector, state_dict, remove_keys=[]):
        """
        Convert a vector back to a state dictionary, removing specified keys.

        Args:
            vector (Tensor): The vector to convert.
            state_dict (dict): The reference state dictionary.
            remove_keys (list): List of keys to remove from the state dictionary.

        Returns:
            dict: A state dictionary representation of the vector.
        """
        # create a reference dict to define the order of the vector
        reference_dict = copy.deepcopy(state_dict)
        for key in remove_keys:
            if key in reference_dict:
                del reference_dict[key]
        sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

        # create a shared state dict using the reference dict
        nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

        # add back the encoder and decoder embedding weights.
        if "transformer.shared.weight" in sorted_reference_dict:
            for key in remove_keys:
                sorted_reference_dict[key] = sorted_reference_dict[
                    "transformer.shared.weight"
                ]
        return sorted_reference_dict

    def taskvector_loss(self, layer_vectors, layer_delta, layer_lamdas) -> torch.Tensor:
        """
        Computes the loss based on delta and task vectors for a specific layer.
        """
        total_loss = 0.0

        layer_vectors_scale = layer_vectors * layer_lamdas.view(-1, 1, 1)
        sum_over_num_vectors = layer_vectors_scale.sum(dim=0)

        layer_delta_scale = layer_delta * layer_lamdas.view(-1, 1, 1)
        sum_over_delta = layer_delta_scale.sum(dim=0)

        # Iterate through each vector and calculate the loss one by one
        for v_j in layer_vectors:
            part1 = -v_j * sum_over_num_vectors
            part2 = -v_j * sum_over_delta
            part3 = v_j * v_j

            expression = part1 + part2 + part3
            layer_loss = expression.sum(dim=1).pow(2).sum()

            # Cumulative total loss
            total_loss += layer_loss
        return total_loss

    def compute_layer_lamdas(self, vectors: List[StateDictType]) -> torch.Tensor:
        lamdas = []
        for vec in vectors:
            tmp = {}
            for layer_name in vec.keys():
                norm_vec = torch.norm(vec[layer_name])
                tmp[layer_name] = 0.07 / norm_vec
            lamdas.append(tmp)
        return lamdas

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
            task_vector_items = list(task_vector.items())
            for w, (name, param) in zip(weight, task_vector_items):
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
