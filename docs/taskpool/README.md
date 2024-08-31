# Introduction to Taskpool Module

A taskpool is a collection of tasks that can be used to evaluate the performance of merged models.
Each task in the taskpool is defined by a dataset and a metric.

A taskpool is specified by a `yaml` configuration file, which often contains the following fields:

- `type`: The type of the taskpool.
- `dataset_type`: The type of the dataset used in the tasks.
- `tasks`: A list of tasks, each task is dict with the following fields:
    - `name`: The name of the task.
    - `dataset`: The dataset used for the task.
    - `metric`: The metric used to evaluate the performance of the model on the task.

### References

::: fusion_bench.taskpool.load_taskpool_from_config

::: fusion_bench.taskpool.TaskPool
