from typing import Any, Callable, Dict, List, cast

import numpy as np
import torch
from omegaconf import ListConfig
from torch import Tensor, nn


def aggregate_tensors(outputs: List[Any], aggregate_fn: Callable) -> Tensor:
    """
    Aggregates a list of outputs using the provided aggregation function.

    This function handles different types of outputs:
    - If the outputs are Tensors, it applies the aggregation function directly.
    - If the outputs are dictionaries, it recursively aggregates each value.
    - If the outputs are tuples or lists, it recursively aggregates each element.
    - If all outputs are None, it returns None.
    - If the outputs are of an unsupported type, it raises a ValueError.

    Args:
        outputs (list): A list of outputs to be aggregated. The outputs can be Tensors, dictionaries, tuples, lists, or None.
        aggregate_fn (callable): A function to aggregate the outputs. Typically, this could be a function like `torch.mean`.

    Returns:
        Tensor or dict or tuple or list or None: The aggregated output, matching the type of the input outputs.

    Raises:
        ValueError: If the outputs are of an unsupported type.
    """
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
    """
    Ensemble module that averages the outputs of multiple models.
    """

    def __init__(self, models: List[nn.Module]):
        """
        Initializes the EnsembleModule with a list of models.

        Args:
            models (List[nn.Module]): List of models to ensemble.
        """
        super().__init__()
        # TODO: distribute models to devices
        self.model_list = nn.ModuleList(models)

    def _aggregate_tensors(self, outputs: List[Tensor]) -> Tensor:
        """
        Aggregates a list of tensors by computing their mean.

        Args:
            outputs (List[Tensor]): List of tensors to aggregate.

        Returns:
            Tensor: The mean tensor.
        """
        return torch.stack(outputs).mean(dim=0)

    def forward(self, *args, **kwargs):
        """
        Performs a forward pass by averaging the outputs of the models.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Aggregated output from the ensemble of models.
        """
        outputs = [model(*args, **kwargs) for model in self.model_list]
        return aggregate_tensors(outputs, self._aggregate_tensors)


class WeightedEnsembleModule(nn.Module):
    """
    Ensemble module that computes a weighted average of the outputs from multiple models.
    """

    def __init__(
        self,
        models: List[nn.Module],
        weights: List[float] | Tensor | np.ndarray,
        normalize: bool = True,
    ):
        """
        Initializes the WeightedEnsembleModule with models and their corresponding weights.

        Args:
            models (List[nn.Module]): List of models to ensemble.
            weights (List[float] | Tensor | np.ndarray): Weights for each model.
            normalize (bool, optional): If True, normalizes the weights. Defaults to True.
        """
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
        """
        Aggregates a list of tensors using the provided weights.

        Args:
            outputs (List[Tensor]): List of tensors to aggregate.

        Returns:
            Tensor: The weighted sum of the tensors.
        """
        weights = cast(Tensor, self.weights).view(-1, *([1] * outputs[0].dim()))
        return (torch.stack(outputs) * weights).sum(dim=0)

    def forward(self, *args, **kwargs):
        """
        Performs a forward pass by computing the weighted average of the models' outputs.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Weighted aggregated output from the ensemble of models.
        """
        outputs = [model(*args, **kwargs) for model in self.model_list]
        return aggregate_tensors(outputs, self._aggregate_tensors)


class MaxModelPredictor(EnsembleModule):
    """
    Ensemble module that selects the maximum output among multiple models.
    """

    def _aggregate_tensors(self, outputs: List[Tensor]) -> Tensor:
        """
        Aggregates a list of tensors by selecting the maximum value at each position.

        Args:
            outputs (List[Tensor]): List of tensors to aggregate.

        Returns:
            Tensor: Tensor with the maximum values.
        """
        return torch.stack(outputs).max(dim=0).values
