import logging
from abc import ABC, abstractmethod
from typing import Union

from omegaconf import DictConfig
from tqdm.autonotebook import tqdm

from fusion_bench.mixins import BaseYAMLSerializableModel


class BaseTaskPool(BaseYAMLSerializableModel):
    _program = None

    @abstractmethod
    def evaluate(self, model):
        """
        Evaluate the model on all tasks in the task pool, and return a report.

        Take image classification as an example, the report will look like:

        ```python
        {
            "mnist": {
                "accuracy": 0.8,
                "loss": 0.2,
            },
            <task_name>: {
                <metric_name>: <metric_value>,
                ...
            },
        }
        ```

        Args:
            model: The model to evaluate.

        Returns:
            report (dict): A dictionary containing the results of the evaluation for each task.
        """
        pass
