import os

import lightning as L
import optuna
from hydra import compose, initialize

from fusion_bench import instantiate
from fusion_bench.method import TaskArithmeticAlgorithm
from fusion_bench.modelpool import CLIPVisionModelPool
from fusion_bench.scripts.cli import _get_default_config_path
from fusion_bench.taskpool import CLIPVisionModelTaskPool

fabric = L.Fabric(accelerator="auto", devices=1)

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


def average_accuracy(trail: optuna.Trial) -> float:
    scaling_factor = trail.suggest_float("x", 0.0, 1.0)
    algorithm = TaskArithmeticAlgorithm(scaling_factor=scaling_factor)
    merged_model = algorithm.run(modelpool)
    report = taskpool.evaluate(merged_model)
    return report["average"]["accuracy"]


study = optuna.create_study(
    storage="sqlite:///hyperparam_search.db",
    study_name="arithmetic_task_on_eight_clip_models",
    direction=optuna.study.StudyDirection.MAXIMIZE,
)
study.optimize(average_accuracy, n_trials=20)
print(f"Best value: {study.best_value} (params: {study.best_params})")
