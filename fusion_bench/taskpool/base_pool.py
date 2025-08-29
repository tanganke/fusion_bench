from abc import abstractmethod
from typing import Any, Dict

from fusion_bench.mixins import BaseYAMLSerializable


class BaseTaskPool(BaseYAMLSerializable):
    """Abstract base class for task pools in the FusionBench framework.

    A task pool represents a collection of evaluation tasks that can be used to
    assess model performance across multiple benchmarks or datasets. This base
    class defines the common interface that all task pool implementations must
    follow, ensuring consistency across different task types and evaluation
    scenarios.

    Task pools are designed to be configurable through YAML files and can be
    used in various model fusion and evaluation workflows. They provide a
    standardized way to evaluate models on multiple tasks and aggregate results.

    The class inherits from BaseYAMLSerializable to support configuration
    management and serialization capabilities.

    Attributes:
        _program: Optional program reference for execution context.
        _config_key: Configuration key used for YAML configuration ("taskpool").

    Abstract Methods:
        evaluate: Must be implemented by subclasses to define task-specific
            evaluation logic.

    Example:
        Implementing a custom task pool:

        ```python
        class MyTaskPool(BaseTaskPool):


            def evaluate(self, model, **kwargs):
                results = {}
                for task_name in self.tasks:
                    # Implement task-specific evaluation
                    results[task_name] = self._evaluate_task(model, task_name)
                return results
        ```
    """

    _program = None
    _config_key = "taskpool"

    @abstractmethod
    def evaluate(self, model: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Evaluate a model on all tasks in the task pool and return aggregated results.

        This abstract method defines the core evaluation interface that all task pool
        implementations must provide. It should evaluate the given model on all tasks
        managed by the pool and return a structured report of the results.

        The evaluation process typically involves:
        1. Iterating through all tasks in the pool
        2. Running model inference on each task's dataset
        3. Computing task-specific metrics
        4. Aggregating results into a standardized report format

        Args:
            model: The model to evaluate. Can be any model type (PyTorch model,
                Hugging Face model, etc.) that is compatible with the specific
                task pool implementation.
            *args: Additional positional arguments that may be needed for
                task-specific evaluation procedures.
            **kwargs: Additional keyword arguments for evaluation configuration,
                such as batch_size, device, evaluation metrics, etc.

        Returns:
            Dict[str, Any]: A dictionary containing evaluation results for each task.
                The structure follows the pattern:

                ```python
                {
                    "task_name_1": {
                        "metric_1": value,
                        "metric_2": value,
                        ...
                    },
                    "task_name_2": {
                        "metric_1": value,
                        "metric_2": value,
                        ...
                    },
                    ...
                }
                ```

        Example:
            For an image classification task pool:

            ```python
            results = task_pool.evaluate(model)
            # Returns:
            # {
            #     "mnist": {
            #         "accuracy": 0.95,
            #         "loss": 0.15,
            #     },
            #     "cifar10": {
            #         "accuracy": 0.87,
            #         "loss": 0.42,
            #     }
            # }
            ```

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        Note:
            Implementations should ensure that the returned dictionary structure
            is consistent and that metric names are standardized across similar
            task types to enable meaningful comparison and aggregation.
        """
        pass
