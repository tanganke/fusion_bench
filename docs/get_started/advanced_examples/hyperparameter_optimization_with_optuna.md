# Hyperparameter Optimization with Optuna

This guide demonstrates how to use [Optuna](https://optuna.org/) for hyperparameter optimization in FusionBench. We'll walk through optimizing the scaling factor for the Task Arithmetic algorithm using automated hyperparameter search.

## Overview

Hyperparameter optimization is crucial for achieving optimal model fusion performance. Instead of manually testing different parameter combinations, Optuna provides intelligent search strategies to find the best hyperparameters efficiently.

The example shows how to optimize the `scaling_factor` parameter of the [`TaskArithmeticAlgorithm`][fusion_bench.method.TaskArithmeticAlgorithm] across multiple CLIP models.

## Prerequisites

Install Optuna if you haven't already:

```bash
pip install optuna
pip install optuna-dashboard
```

## Implementation

The complete implementation can be found in [examples/hyperparam_search/task_arithmetic.py](https://github.com/tanganke/fusion_bench/tree/main/examples/hyperparam_search/task_arithmetic.py):

### 1. Setup and Configuration

```python
import os
import lightning as L
import optuna
from hydra import compose, initialize

from fusion_bench import instantiate
from fusion_bench.method import TaskArithmeticAlgorithm
from fusion_bench.modelpool import CLIPVisionModelPool
from fusion_bench.scripts.cli import _get_default_config_path
from fusion_bench.taskpool import CLIPVisionModelTaskPool

# Initialize Lightning Fabric for efficient computation
fabric = L.Fabric(accelerator="auto", devices=1)

# Load configuration using Hydra
with initialize(
    version_base=None,
    config_path=os.path.relpath(
        _get_default_config_path(), start=os.path.dirname(__file__)
    ),
):
    cfg = compose(
        config_name="fabric_model_fusion",
        overrides=[
            "modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8",
            "taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8",
        ],
    )
    modelpool: CLIPVisionModelPool = instantiate(cfg.modelpool)
    taskpool: CLIPVisionModelTaskPool = instantiate(cfg.taskpool)
    taskpool._fabric_instance = fabric
```

### 2. Define the Objective Function

The objective function evaluates model performance for a given set of hyperparameters:

```python
def average_accuracy(trial: optuna.Trial) -> float:
    # Suggest a scaling factor value between 0.0 and 1.0
    scaling_factor = trial.suggest_float("x", 0.0, 1.0)
    
    # Create algorithm with the suggested hyperparameter
    algorithm = TaskArithmeticAlgorithm(scaling_factor=scaling_factor)
    
    # Run model fusion
    merged_model = algorithm.run(modelpool)
    
    # Evaluate the merged model
    report = taskpool.evaluate(merged_model)
    
    # Return the metric to optimize (average accuracy across tasks)
    return report["average"]["accuracy"]
```

### 3. Run Optimization Study

```python
# Create an Optuna study with SQLite storage for persistence
study = optuna.create_study(
    storage="sqlite:///hyperparam_search.db",
    study_name="arithmetic_task_on_eight_clip_models",
    direction=optuna.study.StudyDirection.MAXIMIZE,
)

# Optimize for 20 trials
study.optimize(average_accuracy, n_trials=20)

# Print the best results
print(f"Best value: {study.best_value} (params: {study.best_params})")
```

The example uses SQLite storage (`sqlite:///hyperparam_search.db`) to persist optimization results. This allows you to:

- Resume interrupted studies
- Analyze results later
- Share studies across different runs

## Running the Example

Execute the hyperparameter optimization:

```bash
cd examples/hyperparam_search
python task_arithmetic.py
```

Launch the Optuna dashboard to visualize optimization results:

```bash
optuna-dashboard sqlite:///hyperparam_search.db
```

This opens a web interface (typically at `http://localhost:8080`) where you can:

- View optimization history and parameter importance
- Analyze trial performance with interactive plots
- Compare different hyperparameter combinations
- Monitor convergence and identify optimal regions

The dashboard provides insights into how different scaling factors affect model performance, helping you understand the hyperparameter landscape.
