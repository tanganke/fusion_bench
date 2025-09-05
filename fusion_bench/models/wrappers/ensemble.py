import logging
from typing import Any, Callable, Dict, Generic, List, Union, cast

import numpy as np
import torch
import torch.futures
from omegaconf import ListConfig
from torch import Tensor, nn

from fusion_bench.utils.devices import to_device
from fusion_bench.utils.type import TorchModelType

log = logging.getLogger(__name__)


def aggregate_tensors(
    outputs: List[Any], aggregate_fn: Callable
) -> Union[Tensor, Dict, List, None]:
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


class EnsembleModule(nn.Module, Generic[TorchModelType]):
    """
    Ensemble module that averages the outputs of multiple models.
    """

    def __init__(
        self,
        models: List[TorchModelType],
        device_map: Dict[int, Union[int, str]] | None = None,
    ):
        """
        Initializes the EnsembleModule with a list of models.

        Args:
            models (List[nn.Module]): List of models to ensemble.
        """
        super().__init__()
        # TODO: distribute models to devices
        self.model_list = nn.ModuleList(models)
        self.device_map = device_map
        if self.device_map is not None:
            self._move_models_to_devices()

    def _move_models_to_devices(self):
        for model_idx, device_id in self.device_map.items():
            log.info(f"Moving model {model_idx} to device {device_id}")
            self.model_list[model_idx] = self.model_list[model_idx].to(
                device_id, non_blocking=True
            )

    def _aggregate_tensors(self, outputs: List[Tensor]) -> Tensor:
        """
        Aggregates a list of tensors by computing their mean.

        Args:
            outputs (List[Tensor]): List of tensors to aggregate.

        Returns:
            Tensor: The mean tensor.
        """
        return torch.stack(outputs).mean(dim=0)

    def _parallel_forward_with_device_map(self, *args: Any, **kwargs: Any) -> List[Any]:
        """
        Performs parallel forward pass using device mapping with futures.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[Any]: List of outputs from all models, all moved to the same device.
        """
        futures = []

        device_data_cache = {}
        for i, model in enumerate(self.model_list):
            device_id = self.device_map.get(i, "cpu")

            if device_id not in device_data_cache:
                # Move inputs to the same device as the model
                device_args = to_device(
                    args, device_id, copy_on_move=True, non_blocking=True
                )
                device_kwargs = to_device(
                    kwargs, device_id, copy_on_move=True, non_blocking=True
                )
                device_data_cache[device_id] = (device_args, device_kwargs)
            else:
                device_args, device_kwargs = device_data_cache[device_id]

            # Create a future for asynchronous execution
            future = torch.jit.fork(model, *device_args, **device_kwargs)
            futures.append(future)

        # Wait for all futures to complete and collect results
        outputs = [torch.jit.wait(future) for future in futures]

        # Move all outputs to the same device (use the device of the first model or cpu as fallback)
        target_device = self.device_map.get(0, "cpu") if self.device_map else "cpu"
        outputs = [
            to_device(output, target_device, non_blocking=True) for output in outputs
        ]
        return outputs

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Performs a forward pass by averaging the outputs of the models.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Aggregated output from the ensemble of models.
        """
        if self.device_map is None:
            outputs = [model(*args, **kwargs) for model in self.model_list]
        else:
            # Parallel execution with device mapping
            outputs = self._parallel_forward_with_device_map(*args, **kwargs)
        return aggregate_tensors(outputs, self._aggregate_tensors)


class WeightedEnsembleModule(nn.Module, Generic[TorchModelType]):
    """
    Ensemble module that computes a weighted average of the outputs from multiple models.
    """

    def __init__(
        self,
        models: List[TorchModelType],
        weights: List[float] | Tensor | np.ndarray,
        normalize: bool = True,
        device_map: Dict[int, Union[int, str]] | None = None,
    ):
        """
        Initializes the WeightedEnsembleModule with models and their corresponding weights.

        Args:
            models (List[nn.Module]): List of models to ensemble.
            weights (List[float] | Tensor | np.ndarray): Weights for each model.
            normalize (bool, optional): If True, normalizes the weights. Defaults to True.
            device_map (Dict[int, Union[int, str]] | None, optional): Device mapping for parallel execution. Defaults to None.
        """
        super().__init__()
        self.model_list = nn.ModuleList(models)
        self.device_map = device_map

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

        if self.device_map is not None:
            self._move_models_to_devices()

    def _move_models_to_devices(self):
        """Move models to their assigned devices according to device_map."""
        for model_idx, device_id in self.device_map.items():
            log.info(f"Moving model {model_idx} to device {device_id}")
            self.model_list[model_idx] = self.model_list[model_idx].to(
                device_id, non_blocking=True
            )

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

    def _parallel_forward_with_device_map(self, *args: Any, **kwargs: Any) -> List[Any]:
        """
        Performs parallel forward pass using device mapping with futures.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[Any]: List of outputs from all models, all moved to the same device.
        """
        futures = []

        device_data_cache = {}
        for i, model in enumerate(self.model_list):
            device_id = self.device_map.get(i, "cpu")

            if device_id not in device_data_cache:
                # Move inputs to the same device as the model
                device_args = to_device(
                    args, device_id, copy_on_move=True, non_blocking=True
                )
                device_kwargs = to_device(
                    kwargs, device_id, copy_on_move=True, non_blocking=True
                )
                device_data_cache[device_id] = (device_args, device_kwargs)
            else:
                device_args, device_kwargs = device_data_cache[device_id]

            # Create a future for asynchronous execution
            future = torch.jit.fork(model, *device_args, **device_kwargs)
            futures.append(future)

        # Wait for all futures to complete and collect results
        outputs = [torch.jit.wait(future) for future in futures]

        # Move all outputs to the same device (use the device of the first model or cpu as fallback)
        target_device = self.device_map.get(0, "cpu") if self.device_map else "cpu"
        outputs = [to_device(output, target_device) for output in outputs]

        return outputs

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Performs a forward pass by computing the weighted average of the models' outputs.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Weighted aggregated output from the ensemble of models.
        """
        if self.device_map is None:
            outputs = [model(*args, **kwargs) for model in self.model_list]
        else:
            # Parallel execution with device mapping
            outputs = self._parallel_forward_with_device_map(*args, **kwargs)
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
