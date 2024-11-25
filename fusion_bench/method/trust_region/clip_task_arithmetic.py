"""
Implementation of Task Arithmetic in Trust Region: A Training-Free Model Merging Approach to Navigate Knowledge Conflicts
https://openreview.net/forum?id=q3ztjJRQuJ
"""

import logging
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Iterable, List, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing_extensions import override

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.dataset.clip_dataset import CLIPDataset
from fusion_bench.mixins import CLIPClassificationMixin, SimpleProfilerMixin
from fusion_bench.utils import first
from fusion_bench.utils.state_dict_arithmetic import state_dict_sub
from fusion_bench.utils.type import StateDictType

from .utils import state_dict_to_vector, vector_to_state_dict

log = logging.getLogger(__name__)


def trainable_state_dict(module: nn.Module) -> StateDictType:
    """
    Returns the state dictionary of the module containing only the trainable parameters.

    Args:
        module (nn.Module): The neural network module.

    Returns:
        Dict[str, Tensor]: A dictionary containing the names and values of the trainable parameters.
    """
    return {
        name: param for name, param in module.named_parameters() if param.requires_grad
    }


class TaskArithmeticWithTrustRegionForCLIP(
    BaseAlgorithm,
    SimpleProfilerMixin,
    CLIPClassificationMixin,
):
    def __init__(
        self,
        scaling_factor: Union[float, List[float]],
        threshold_quantile: float,
        max_samples: int,
        batch_size: int,
        zero_shot: bool,
        **kwargs,
    ):
        self.scaling_factor = scaling_factor
        self.threshold_quantile = threshold_quantile
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.zero_shot = zero_shot
        super().__init__(**kwargs)

    @override
    def run(self, modelpool: BaseModelPool):
        self.modelpool = modelpool

        # compute the task vectors
        pretrained_model, task_vectors = self.compute_vanilla_task_vectors()
        task_vectors = {
            name: state_dict_to_vector(task_vector)
            for name, task_vector in task_vectors.items()
        }

        if not self.zero_shot:
            all_avg_abs_grads = self.compute_avg_abs_grads(pretrained_model)
            all_avg_abs_grads = {
                n: state_dict_to_vector(grad) for n, grad in all_avg_abs_grads.items()
            }
        else:
            # the task vector is used to estimate the gradient
            all_avg_abs_grads = {name: tv.abs() for name, tv in task_vectors.items()}

        # compute the trust region
        Omega = torch.zeros_like(first(all_avg_abs_grads.values()))

        for i in all_avg_abs_grads:
            for j in all_avg_abs_grads:
                if i != j:
                    vector1 = all_avg_abs_grads[i]
                    vector2 = torch.abs(task_vectors[j])
                    Omega += vector1 * vector2

        values, indices = Omega.sort(descending=False)
        threshold = values[
            max(0, min(int(Omega.numel() * self.threshold_quantile), Omega.numel() - 1))
        ]

        mask = (Omega < threshold).bool()

        # compute the task vectors
        for task in task_vectors:
            task_vectors[task] = task_vectors[task] * mask

        task_vector_sum = sum(task_vectors.values())
        task_vector_sum = vector_to_state_dict(
            task_vector_sum, trainable_state_dict(pretrained_model)
        )

        if isinstance(self.scaling_factor, (int, float)):
            model = pretrained_model
            for name, param in model.named_parameters():
                param.data += task_vector_sum[name] * self.scaling_factor
            return model
        elif isinstance(self.scaling_factor, Iterable):
            models = {}
            for scaling_factor in self.scaling_factor:
                model = deepcopy(pretrained_model)
                for name, param in pretrained_model.named_parameters():
                    param.data += task_vector_sum[name] * scaling_factor
                models[scaling_factor] = model
            return models
        else:
            raise ValueError(
                f"Incorrect type of `scaling_factor`: {type(self.scaling_factor)}. "
                "It should be a single real number or a list of real numbers."
            )

    def compute_avg_abs_grads(self, pretrained_model):
        modelpool = self.modelpool

        self.setup_zero_shot_classification_head()

        pretrained_model = (
            deepcopy(pretrained_model)
            if pretrained_model is not None
            else modelpool.load_pretrained_model()
        )
        pretrained_model = self.fabric.setup_module(pretrained_model)
        pretrained_model.train()

        all_avg_abs_grads: Dict[str, StateDictType] = {}
        for train_dataset_name in (
            pbar := tqdm(
                modelpool.train_dataset_names, desc="Train datasets", dynamic_ncols=True
            )
        ):
            pbar.set_description(f"Train dataset: {train_dataset_name}")
            dataset = modelpool.load_train_dataset(train_dataset_name)
            dataset = CLIPDataset(dataset, self.clip_processor)
            dataloader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)
            dataloader = self.fabric.setup_dataloaders(dataloader)

            grad: StateDictType = defaultdict(float)
            num_samples = 0
            for batch in dataloader:
                images, labels = batch
                batch_size = images.size(0)

                if num_samples + batch_size > self.max_samples:
                    batch_size = self.max_samples - num_samples
                    images = images[:batch_size]
                    labels = labels[:batch_size]

                logits = self.compute_logits(
                    pretrained_model, images, task=train_dataset_name
                )
                for i in range(batch_size):
                    pretrained_model.zero_grad()
                    loss = F.cross_entropy(logits[i], labels[i])
                    self.fabric.backward(
                        loss, retain_graph=True if i != batch_size - 1 else False
                    )
                    for name, param in pretrained_model.module.named_parameters():
                        if param.requires_grad:
                            grad[name] += torch.abs(param.grad).detach()

                num_samples += batch_size
                if num_samples >= self.max_samples:
                    break

            for name in grad:
                grad[name] = (grad[name] / num_samples).cpu()

            all_avg_abs_grads[name] = grad
        return all_avg_abs_grads

    @torch.no_grad()
    def compute_vanilla_task_vectors(self):
        modelpool = self.modelpool

        pretrained_model = modelpool.load_pretrained_model()
        pretrained_sd = trainable_state_dict(pretrained_model)
        finetuned_sds = {
            name: trainable_state_dict(model)
            for name, model in modelpool.named_models()
        }

        task_vectors = {
            name: state_dict_sub(finetuned, pretrained_sd)
            for name, finetuned in finetuned_sds.items()
        }
        return pretrained_model, task_vectors
