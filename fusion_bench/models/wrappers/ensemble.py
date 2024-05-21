from typing import Dict, List, cast

import numpy as np
import torch
from omegaconf import ListConfig
from torch import Tensor, nn


def aggregate_tensors(outputs, aggregate_fn) -> Tensor:
    # If the output is a Tensor, take the mean
    if isinstance(outputs[0], torch.Tensor):
        return aggregate_fn(outputs)

    # If the output is a dict, take the mean of each value
    elif isinstance(outputs[0], Dict):
        result = type(outputs[0])()
        for key in outputs[0]:
            result[key] = aggregate_tensors(
                [output[key] for output in outputs], aggregate_fn
            )
        return result

    # If the output is a tuple or list, take the mean of each element
    elif isinstance(outputs[0], (tuple, list)):
        return tuple(
            aggregate_tensors([output[i] for output in outputs], aggregate_fn)
            for i in range(len(outputs[0]))
        )

    # If the output is None, return None
    elif all(output is None for output in outputs):
        return None

    # If the output is none of the above, return as is
    else:
        raise ValueError("Unsupported type for outputs")


class EnsembleModule(nn.Module):
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        # TODO: distribute models to devices
        self.model_list = nn.ModuleList(models)

    def _aggregate_tensors(self, outputs: List[Tensor]) -> Tensor:
        return torch.stack(outputs).mean(dim=0)

    def forward(self, *args, **kwargs):
        outputs = [model(*args, **kwargs) for model in self.model_list]
        return aggregate_tensors(outputs, self._aggregate_tensors)


class WeightedEnsembleModule(nn.Module):
    def __init__(
        self,
        models: List[nn.Module],
        weights: List[float] | Tensor | np.ndarray,
        normalize: bool = True,
    ):
        super().__init__()
        self.model_list = nn.ModuleList(models)
        if isinstance(weights, (list, tuple, ListConfig)):
            weights = torch.tensor(weights)
        elif isinstance(weights, Tensor):
            weights = weights
        elif isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights)
        else:
            raise ValueError(f"Unsupported type for weights: {type(weights)=}")

        assert len(models) == len(weights) and weights.dim() == 1, (
            "weights must be a 1D tensor of the same length as models."
            f"But got {len(models)=}, {weights.dim()=}"
        )
        if normalize:
            weights = weights / weights.sum()
        self.register_buffer("weights", weights)

    def _aggregate_tensors(self, outputs: List[Tensor]) -> Tensor:
        weights = cast(Tensor, self.weights).view(-1, *([1] * outputs[0].dim()))
        return (torch.stack(outputs) * weights).sum(dim=0)

    def forward(self, *args, **kwargs):
        outputs = [model(*args, **kwargs) for model in self.model_list]
        return aggregate_tensors(outputs, self._aggregate_tensors)


class MaxModelPredictor(EnsembleModule):
    def _aggregate_tensors(self, outputs: List[Tensor]) -> Tensor:
        return torch.stack(outputs).max(dim=0).values
