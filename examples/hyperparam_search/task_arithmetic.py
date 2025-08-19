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
