from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional

import torch
from torch import nn

from fusion_bench import TorchModelType


class ModulatedModel(nn.Module, Generic[TorchModelType]):
    """
    A model wrapper that uses task-specific modulators to adapt a shared backbone
    for different tasks.

    The model maintains a shared backbone and task-specific modulators. During forward pass,
    the appropriate modulator is applied based on the current task.
    """

    _current_task: Optional[str] = None

    def __init__(
        self,
        backbone: TorchModelType,
        modulators: Dict[str, "TaskModulator[TorchModelType]"],
    ):
        super().__init__()
        self.backbone = backbone
        self.modulators = nn.ModuleDict(modulators)

    def add_modulator(self, task_name: str, modulator: "TaskModulator[TorchModelType]"):
        """Add a new task-specific modulator."""
        self.modulators[task_name] = modulator

    def set_task(self, task_name: str):
        """Set the current task for inference."""
        if task_name not in self.modulators:
            raise ValueError(
                f"Task '{task_name}' not found in modulators. Available tasks: {list(self.modulators.keys())}"
            )
        self._current_task = task_name

    @property
    def current_task(self) -> Optional[str]:
        """Get the current task name."""
        return self._current_task

    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass with task-specific modulation.

        Args:
            *args: Positional arguments for the backbone model
            task: Task name to use (overrides current_task if provided)
            **kwargs: Keyword arguments for the backbone model

        Returns:
            Model output after applying task-specific modulation
        """
        if self._current_task is None and "task" not in kwargs:
            raise ValueError(
                "No task specified. Set current_task or provide 'task' argument."
            )


class TaskModulator(nn.Module, Generic[TorchModelType], ABC):
    """
    Lightweight, task-specific parameterization that modulates
    a shared representation.

    This is the base class for all task modulators. Subclasses should implement
    the `apply` method to define how the modulator adapts the backbone model
    for a specific task.
    """

    @abstractmethod
    def apply(self, modulated_model: "ModulatedModel[TorchModelType]"):
        """
        Apply task-specific modulation to the backbone model.

        Args:
            backbone: The shared backbone model

        Returns:
            Model output after applying task-specific modulation
        """
        raise NotImplementedError("Subclasses must implement the apply method.")
