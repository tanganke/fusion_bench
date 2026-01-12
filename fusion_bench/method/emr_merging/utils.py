from typing import Dict

import numpy as np
import torch
from torch import nn

from fusion_bench import StateDictType, TorchModelType
from fusion_bench.models.modulator import ModulatedModel, TaskModulator
from fusion_bench.models.modulator.base import ModulatedModel, TaskModulator
from fusion_bench.models.parameter_dict import ParameterDictModel
from fusion_bench.utils.state_dict_arithmetic import state_dict_sum


def _sign(x: torch.Tensor) -> torch.Tensor:
    """
    Return the sign of the tensor: 1 for positive, -1 for negative.
    Zeros are treated as negative (i.e., sign  -1).
    """
    return (x > 0) * 2 - 1


def emr_merge(task_vectors: list[StateDictType]):
    """
    Modified from original EMR merging function to return unified vector, masks, and rescalers.

    Args:
        task_vectors: List of task-specific vectors (state dicts).

    Returns:
        vector_unified: The unified task vector (state dict).
        masks: Dict mapping parameter names to list of task-specific masks (tensors).
        rescalers: Tensor of rescaling factors for each task.
    """
    num_tasks = len(task_vectors)

    # compute the sign flag
    # \gamma_uni = sign( sum_i tau_i )
    flag_dict = {k: _sign(v) for k, v in state_dict_sum(task_vectors).items()}

    # \tau_uni
    vector_unified = {}
    scales = torch.zeros(num_tasks)
    # mask indicate if the direction of the task vector aligns with the unified vector
    # {<param_name>: [mask_task1, mask_task2, ...]}
    masks: dict[str, list[torch.Tensor]] = {}
    for n, flag in flag_dict.items():
        masks[n] = []
        param_max = torch.zeros_like(task_vectors[0][n])
        for m in range(num_tasks):
            param = task_vectors[m][n]
            mask = (param * flag) > 0
            masks[n].append(mask)
            param_abs = torch.abs(mask * param)
            param_max = torch.where(param_abs > param_max, param_abs, param_max)
            scales[m] += torch.mean(torch.abs(param))
        vector_unified[n] = param_max * flag

    new_scales = torch.zeros(num_tasks)
    for m in range(num_tasks):
        for n in vector_unified:
            p = vector_unified[n] * masks[n][m]
            new_scales[m] += torch.mean(torch.abs(p))
    rescalers = scales / new_scales

    return vector_unified, masks, rescalers


class EMRModulatedModel(ModulatedModel[TorchModelType]):
    """
    Modulated Model for EMR Merging.
    """

    def __init__(
        self,
        backbone: TorchModelType,
        modulators: Dict[str, "EMRTaskModulator"],
        unified_task_vector: StateDictType,
    ):
        super().__init__(backbone, modulators)

        unified_task_vector = unified_task_vector.copy()
        for name, tensor in unified_task_vector.items():
            if not isinstance(tensor, (nn.Parameter, nn.Buffer)):
                unified_task_vector[name] = nn.Parameter(tensor, requires_grad=False)
        self.unified_task_vector = ParameterDictModel(unified_task_vector)


class EMRTaskModulator(TaskModulator[TorchModelType]):
    """
    Task Modulator for EMR (Elect, Mask & Rescale) Merging.

    This modulator applies task-specific adaptations to a unified model by:
    1. Masking: Aligning direction with task-specific model (mask sets inconsistent signs to zero)
    2. Rescaling: Aligning magnitude with task-specific model

    The application formula is:
        θ_new = θ_old + τ_unified ⊙ mask_i * rescaler_i

    where:
        - τ_unified is the unified task vector (elected from all task vectors)
        - mask_i is the task-specific binary mask
        - rescaler_i is the task-specific rescaling factor

    Args:
        vector: The unified task vector (τ_unified) as a state dict
        mask: Task-specific binary mask as a dict of tensors
        rescaler: Task-specific rescaling factor (scalar)
    """

    def __init__(
        self,
        mask: Dict[str, torch.Tensor],
        rescaler: float,
    ):
        super().__init__()

        # Store masks separately with a prefix to avoid conflicts
        mask = mask.copy()
        for name, tensor in mask.items():
            if not isinstance(tensor, (nn.Parameter, nn.Buffer)):
                mask[name] = nn.Parameter(tensor, requires_grad=False)
        self.mask = ParameterDictModel(mask)

        # Register rescaler as a parameter for proper device handling
        self.rescaler = nn.Parameter(torch.tensor(rescaler), requires_grad=False)

    @torch.no_grad()
    def apply(self, modulated_model: "EMRModulatedModel[TorchModelType]"):
        """
        Apply the EMR task vector to the model.

        For each parameter in the state dict:
            θ_new = θ_old + τ_unified ⊙ mask_i * rescaler_i

        This applies the masked and rescaled unified task vector to align the backbone
        with the task-specific model.
        """
        unified_vector = modulated_model.unified_task_vector

        for name in unified_vector:
            delta = unified_vector[name] * self.mask[name] * self.rescaler
            param = modulated_model.backbone.get_parameter(name)
            param.add_(delta)

    @torch.no_grad()
    def remove(self, modulated_model: "EMRModulatedModel[TorchModelType]"):
        """
        Remove the EMR task vector from the model.

        For each parameter in the state dict:
            θ_old = θ_new - τ_unified ⊙ mask_i * rescaler_i

        This reverses the task-specific adaptation to restore the original backbone.
        """
        unified_vector = modulated_model.unified_task_vector

        for name in unified_vector:
            delta = unified_vector[name] * self.mask[name] * self.rescaler
            param = modulated_model.backbone.get_parameter(name)
            param.sub_(delta)

        modulated_model._current_task = None
