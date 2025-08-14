from abc import abstractmethod
from typing import Any, Dict

from fusion_bench.mixins import BaseYAMLSerializableModel


class BaseTaskPool(BaseYAMLSerializableModel):
    _program = None
    _config_key = "taskpool"

    @abstractmethod
    def evaluate(self, model: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
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
