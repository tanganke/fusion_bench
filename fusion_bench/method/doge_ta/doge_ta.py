R"""
This script contains the general implementation of Modeling Multi-Task Model Merging as Adaptive Projective Gradient Descent.

https://arxiv.org/abs/2501.01230

Example Usage:

```bash
fusion_bench \
    method=doge_ta/doge_ta \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8

fusion_bench \
    method=adamerging \
    method.name=clip_layer_wise_adamerging_doge_ta \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```
"""

import copy
import logging
import time
from collections import OrderedDict
from copy import deepcopy
from functools import reduce
from typing import Dict, List, Mapping, TypeVar, Union  # noqa: F401

import lightning as L
import torch
from torch import nn

from fusion_bench.method.base_algorithm import BaseAlgorithm
from fusion_bench.mixins.lightning_fabric import LightningFabricMixin
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
)
from fusion_bench.utils.type import StateDictType

log = logging.getLogger(__name__)


class DOGE_TA_Algorithm(
    BaseAlgorithm,
    SimpleProfilerMixin,
    LightningFabricMixin,
):
    """
    Task Arithmetic Algorithm for model fusion with learnable delta.

    This class extends the Task Arithmetic method to include a learnable delta
    for task vectors, optimized to maximize cosine similarity among the task vectors.

    Attributes:
        scaling_factor (int): The factor by which the task vectors will be scaled before merging.
        delta (StateDictType): A learnable parameter to adjust task vectors, initialized as zeros.
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "subspace": "subspace",
        "K": "K",
        "lamda": "lamda",
    }

    def __init__(self, subspace, K, lamda):
        self.delta = None  # Initialize delta as None; will be set during run
        self.subspace = subspace
        self.K = K
        self.lamda = lamda
        super().__init__()

    @property
    def device(self) -> torch.device:
        return self.fabric.device

    @torch.no_grad()
    def compute_task_vectors(
        self, modelpool: BaseModelPool, pretrained_model: nn.Module
    ) -> List[StateDictType]:
        """
        Computes task vectors for each model in the model pool relative to the pretrained model.
        """
        task_vectors = []
        pretrained_sd = pretrained_model.state_dict(keep_vars=True)
        filtered_keys = [
            k
            for k in pretrained_sd.keys()
            if ("encoder" in k and "layer_norm" not in k and "weight" in k)
        ]  # Flan T5: "layer_norm" not in k and ("q.weight" in k or "v.weight" in k)

        for model_name in modelpool.model_names:
            model = modelpool.load_model(model_name)
            model_sd = model.state_dict(keep_vars=True)

            filtered_task_vector = {
                k: (model_sd[k] - pretrained_sd[k]) for k in filtered_keys
            }
            task_vectors.append(filtered_task_vector)

        return task_vectors

    def taskvector_loss(self, layer_vectors, layer_delta, layer_lamdas) -> torch.Tensor:
        """
        Computes the loss based on delta and task vectors for a specific layer.
        """
        total_loss = 0.0

        layer_vectors_scale = layer_vectors * layer_lamdas.view(-1, 1, 1)
        sum_over_num_vectors = layer_vectors_scale.sum(dim=0)

        layer_delta_scale = layer_delta.unsqueeze(0) * layer_lamdas.view(-1, 1, 1)
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

    @torch.enable_grad()
    def optimize_delta(self, task_vectors: List[StateDictType]) -> None:
        """
        Optimizes the delta based on the loss of task vectors.
        """
        if self.delta is None:
            self.delta = {
                k: nn.Parameter(torch.zeros_like(v, device=self.device).detach())
                for k, v in task_vectors[0].items()
            }

        optimizer = torch.optim.Adam(self.delta.values(), lr=1e-4)
        initial_mem = torch.cuda.memory_allocated()
        start_time = time.time()
        for layer_name in task_vectors[0].keys():
            layer_vectors = torch.stack([vec[layer_name] for vec in task_vectors]).to(
                self.device
            )
            layer_lamdas = torch.stack(
                [lamdas[layer_name] for lamdas in self.lamdas]
            ).to(self.device)
            for _ in range(400):
                optimizer.zero_grad()
                loss = self.taskvector_loss(
                    layer_vectors, self.delta[layer_name], layer_lamdas
                )
                self.fabric.backward(loss)
                grad_proj = (
                    self.projection[layer_name] @ self.delta[layer_name].grad.detach()
                )
                self.delta[layer_name].grad.data = self.delta[
                    layer_name
                ].grad.data.sub_(grad_proj)
                optimizer.step()
                self.delta[layer_name].grad = None
        end_time = time.time()
        print(f"Running time: {end_time - start_time} s")
        final_mem = torch.cuda.memory_allocated()
        print(f"Memory usage: {(final_mem - initial_mem) / (1024 ** 2)} MB")
        print("Optimization completed.")

    @torch.no_grad()
    def run(self, modelpool: Union[BaseModelPool, Dict[str, nn.Module]]):
        """
        Runs the Algorithm with learnable delta to fuse models in the given model pool.

        Args:
            modelpool (Union[BaseModelPool, Dict[str, nn.Module]]): The pool of models to fuse.

        Returns:
            nn.Module: The pre-trained model with the merged task vectors after optimizing delta.
        """
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        log.info("Fusing models using DOGE_TA with learnable delta.")
        with self.profile("load model"):
            pretrained_model = modelpool.load_model("_pretrained_")

        task_vectors = self.compute_task_vectors(modelpool, pretrained_model)

        self.lamdas = self.compute_layer_lamdas(task_vectors)
        self.projection = {}
        for layer_name in task_vectors[0].keys():
            for i, vector in enumerate(task_vectors):
                layer_vector = vector[layer_name].to(self.device)
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
            layer_proj = torch.matmul(
                u_u[:, : int(s.shape[0] / self.config.subspace)],
                u_u[:, : int(s.shape[0] / self.config.subspace)].T,
            )
            self.projection[layer_name] = layer_proj

        self.optimize_delta(task_vectors)

        del self.projection
        self.delta = {key: param.detach().cpu() for key, param in self.delta.items()}
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
            vector_mask = self.topk_values_mask(flat_vector, K=self.config.K)
            flat_vectors.append(flat_vector)
            vector_masks.append(vector_mask)
        flat_delta = self.state_dict_to_vector(self.delta)

        adjusted_vectors = [
            self.vector_to_state_dict(
                (flat_vector + flat_delta) * vector_mask, self.delta
            )
            for flat_vector, vector_mask in zip(flat_vectors, vector_masks)
        ]

        for layer_name in adjusted_vectors[0].keys():
            layer_vectors = torch.stack(
                [vec[layer_name] for vec in adjusted_vectors], dim=0
            )
            layer_lamdas = torch.stack(
                [lamdas[layer_name] for lamdas in self.lamdas], dim=0
            )
            layer_vectors_scale = layer_vectors * layer_lamdas.view(-1, 1, 1)
            task_vectors[0][layer_name] = layer_vectors_scale.sum(dim=0)

        final_state_dict = {}
        pretrained_sd = pretrained_model.state_dict(keep_vars=True)
        for k, v in pretrained_sd.items():
            if k in task_vectors[0]:
                final_state_dict[k] = v + task_vectors[0][k]
            else:
                final_state_dict[k] = v

        pretrained_model.load_state_dict(final_state_dict)

        self.print_profile_summary()
        return pretrained_model

    def compute_lamdas(self, vectors: List[StateDictType]) -> torch.Tensor:
        lamdas = []
        for vec in vectors:
            norm_vec = torch.norm(
                torch.cat([param.flatten() for param in vec.values()])
            )
            # norm_vec = sum([torch.norm(param) for param in vec.values()])
            lamdas.append(self.config.lamda / norm_vec)
        print(lamdas)
        return lamdas

    def compute_layer_lamdas(self, vectors: List[StateDictType]) -> torch.Tensor:
        lamdas = []
        for vec in vectors:
            tmp = {}
            for layer_name in vec.keys():
                norm_vec = torch.norm(vec[layer_name])
                tmp[layer_name] = self.config.lamda / norm_vec
            lamdas.append(tmp)
        return lamdas

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
