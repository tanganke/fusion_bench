from typing import Union

from omegaconf import DictConfig
from tqdm.autonotebook import tqdm


class TaskPool:
    """
    A class to manage a pool of tasks for evaluation.
    This is the base class for version 0.1.x, deprecated.
    Use `fusion_bench.taskpool.BaseTaskPool` instead.

    Attributes:
        config (DictConfig): The configuration for the task pool.
        _all_task_names (List[str]): A list of all task names in the task pool.
    """

    _program = None

    def __init__(self, taskpool_config: DictConfig):
        """
        Initialize the TaskPool with the given configuration.

        Args:
            taskpool_config (DictConfig): The configuration for the task pool.
        """
        super().__init__()
        self.config = taskpool_config

        # Check for duplicate task names
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
        """
        Return a list of all task names in the task pool.

        Returns:
            List[str]: A list of all task names.
        """
        return self._all_task_names

    def get_task_config(self, task_name: str):
        """
        Retrieve the configuration for a specific task from the task pool.

        Args:
            task_name (str): The name of the task for which to retrieve the configuration.

        Returns:
            DictConfig: The configuration dictionary for the specified task.

        Raises:
            ValueError: If the specified task is not found in the task pool.
        """
        for task in self.config["tasks"]:
            if task["name"] == task_name:
                return task
        raise ValueError(f"Task {task_name} not found in the task pool")

    def load_task(self, task_name_or_config: Union[str, DictConfig]):
        """
        Load a task from the task pool.

        Args:
            task_name_or_config (Union[str, DictConfig]): The name or configuration of the task to load.

        Returns:
            Any: The loaded task.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError
