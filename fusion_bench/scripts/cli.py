"""
This is the CLI script that is executed when the user runs the `fusion-bench` command.
The script is responsible for parsing the command-line arguments, loading the configuration file, and running the fusion algorithm.
"""

import importlib
import importlib.resources
import json
import logging
import os
from typing import Dict, Iterable, Union

import hydra
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm.auto import tqdm

from fusion_bench.method import load_algorithm_from_config
from fusion_bench.mixins.lightning_fabric import LightningFabricMixin
from fusion_bench.modelpool import load_modelpool_from_config
from fusion_bench.taskpool import load_taskpool_from_config
from fusion_bench.utils.rich_utils import print_config_tree

log = logging.getLogger(__name__)


def _get_default_config_path():
    for config_dir in ["fusion_bench_config", "config"]:
        config_path = os.path.join(
            importlib.import_module("fusion_bench").__path__[0], "..", config_dir
        )
        if os.path.exists(config_path) and os.path.isdir(config_path):
            return config_path
    raise FileNotFoundError("Default config path not found.")


def run_model_fusion(cfg: DictConfig):
    """
    Run the model fusion process based on the provided configuration.

    1. This function loads a model pool and an model fusion algorithm based on the configuration.
    2. It then uses the algorithm to fuse the models in the model pool into a single model.
    3. If a task pool is specified in the configuration, it loads the task pool and uses it to evaluate the merged model.
    """
    log.warning(
        "This function is deprecated. Use LightningProgram instead. This will be removed in future versions."
    )
    modelpool = load_modelpool_from_config(cfg.modelpool)

    algorithm = load_algorithm_from_config(cfg.method)
    merged_model = algorithm.run(modelpool)

    if hasattr(cfg, "taskpool") and cfg.taskpool is not None:
        taskpool = load_taskpool_from_config(cfg.taskpool)
        if hasattr(modelpool, "_fabric") and hasattr(taskpool, "_fabric"):
            if taskpool._fabric is None:
                taskpool._fabric = modelpool._fabric
        modelpool.setup_taskpool(taskpool)
        report = taskpool.evaluate(merged_model)
        if cfg.get("save_report", False):
            # save report (Dict) to a file
            # if the directory of `save_report` does not exists, create it
            os.makedirs(os.path.dirname(cfg.save_report), exist_ok=True)
            json.dump(report, open(cfg.save_report, "w"))
    else:
        print("No task pool specified. Skipping evaluation.")


class LightningProgram(LightningFabricMixin):

    def __init__(self, config: DictConfig):
        self.config = config

    def _load_and_setup(self, load_fn, *args, **kwargs):
        """
        Load an object using a provided loading function and setup its attributes.
        """
        obj = load_fn(*args, **kwargs)
        obj._program = self
        if hasattr(obj, "_fabric") and self.fabric is not None:
            obj._fabric = self.fabric
        return obj

    def save_merged_model(self, merged_model):
        if self.config.get("merged_model_save_path", None) is not None:
            # path to save the merged model, use "{log_dir}" to refer to the logger directory
            save_path: str = self.config.merged_model_save_path
            if "{log_dir}" in save_path and self.log_dir is not None:
                save_path = save_path.format(log_dir=self.log_dir)

            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.modelpool.save_model(merged_model, save_path)
        else:
            print("No save path specified for the merged model. Skipping saving.")

    def evaluate_merged_model(
        self, taskpool, merged_modol: Union[nn.Module, Dict, Iterable]
    ):
        """
        Evaluates the merged model using the provided task pool.

        Depending on the type of the merged model, this function handles the evaluation differently:
        - If the merged model is an instance of `nn.Module`, it directly evaluates the model.
        - If the merged model is a dictionary, it extracts the model from the dictionary and evaluates it.
          The evaluation report is then updated with the remaining dictionary items.
        - If the merged model is an iterable, it recursively evaluates each model in the iterable.
        - Raises a `ValueError` if the merged model is of an invalid type.

        Args:
            taskpool: The task pool used for evaluating the merged model.
            merged_modol: The merged model to be evaluated. It can be an instance of `nn.Module`, a dictionary, or an iterable.

        Returns:
            The evaluation report. The type of the report depends on the type of the merged model:
            - If the merged model is an instance of `nn.Module`, the report is a dictionary.
            - If the merged model is a dictionary, the report is a dictionary updated with the remaining dictionary items.
            - If the merged model is an iterable, the report is a list of evaluation reports.
        """
        if isinstance(merged_modol, nn.Module):
            report = taskpool.evaluate(merged_modol)
            print(report)
            return report
        elif isinstance(merged_modol, Dict):
            model = merged_modol.pop("model")
            report: dict = taskpool.evaluate(model)
            report.update(merged_modol)
            print(report)
            return report
        elif isinstance(merged_modol, Iterable):
            return [
                self.evaluate_merged_model(taskpool, m)
                for m in tqdm(merged_modol, desc="Evaluating models")
            ]
        else:
            raise ValueError(f"Invalid type for merged model: {type(merged_modol)}")

    def run_model_fusion(self):
        cfg = self.config

        self.modelpool = modelpool = self._load_and_setup(
            load_modelpool_from_config, cfg.modelpool
        )
        self.alalgorithm = algorithm = self._load_and_setup(
            load_algorithm_from_config, cfg.method
        )
        merged_model = algorithm.run(modelpool)
        self.save_merged_model(merged_model)

        if hasattr(cfg, "taskpool") and cfg.taskpool is not None:
            self.taskpool = taskpool = self._load_and_setup(
                load_taskpool_from_config, cfg.taskpool
            )
            modelpool.setup_taskpool(taskpool)
            report = self.evaluate_merged_model(taskpool, merged_model)
            print(report)
            if cfg.get("save_report", False):
                # save report (Dict) to a file
                # if the directory of `save_report` does not exists, create it
                os.makedirs(os.path.dirname(cfg.save_report), exist_ok=True)
                json.dump(report, open(cfg.save_report, "w"))
        else:
            print("No task pool specified. Skipping evaluation.")


@hydra.main(
    config_path=_get_default_config_path(),
    config_name="example_config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if cfg.print_config:
        print_config_tree(
            cfg,
            print_order=[
                "method",
                "modelpool",
                "taskpool",
            ],
        )
    if cfg.get("dry_run", False):
        log.info("The program is running in dry-run mode. Exiting.")
        return
    if cfg.use_lightning:
        program = LightningProgram(cfg)
        program.run_model_fusion()
    else:
        run_model_fusion(cfg)


if __name__ == "__main__":
    main()
