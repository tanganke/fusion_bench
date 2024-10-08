import logging
from abc import ABC, abstractmethod
from typing import Union

from omegaconf import DictConfig
from tqdm.autonotebook import tqdm


class TaskPool(ABC):
    _program = None

    def __init__(self, taskpool_config: DictConfig):
        super().__init__()
        self.config = taskpool_config

        if self.config.get("tasks", None) is not None:
            task_names = [task["name"] for task in self.config["tasks"]]
            assert len(task_names) == len(
                set(task_names)
            ), "Duplicate task names found in the task pool"
            self._all_task_names = task_names

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
        report = {}
        for task_name in tqdm(self.task_names, desc="Evaluating tasks"):
            task = self.load_task(task_name)
            result = task.evaluate(model)
            report[task_name] = result
        return report

    @property
    def task_names(self):
        return self._all_task_names

    def get_task_config(self, task_name: str):
        for task in self.config["tasks"]:
            if task["name"] == task_name:
                return task
        raise ValueError(f"Task {task_name} not found in the task pool")

    def load_task(self, task_name_or_config: Union[str, DictConfig]):
        raise NotImplementedError
