"""
This is the CLI script that is executed when the user runs the `fusion-bench` command.
The script is responsible for parsing the command-line arguments, loading the configuration file, and running the fusion algorithm.
"""

import importlib
import importlib.resources
import json
import logging
import os

import hydra
import lightning as L
import torch
from lightning.fabric.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from rich import print as rich_print
from rich.syntax import Syntax

from fusion_bench.mixins.lightning_fabric import LightningFabricMixin
from fusion_bench.utils.rich_utils import print_config_tree

from ..method import load_algorithm_from_config
from ..modelpool import load_modelpool_from_config
from ..taskpool import load_taskpool_from_config

log = logging.getLogger(__name__)


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

    def run_model_fusion(self):
        cfg = self.config

        self.modelpool = modelpool = self._load_and_setup(
            load_modelpool_from_config, cfg.modelpool
        )
        self.alalgorithm = algorithm = self._load_and_setup(
            load_algorithm_from_config, cfg.method
        )
        merged_model = algorithm.run(modelpool)

        if hasattr(cfg, "taskpool") and cfg.taskpool is not None:
            self.taskpool = taskpool = self._load_and_setup(
                load_taskpool_from_config, cfg.taskpool
            )
            modelpool.setup_taskpool(taskpool)
            report = taskpool.evaluate(merged_model)
            print(report)
            if cfg.get("save_report", False):
                # save report (Dict) to a file
                # if the directory of `save_report` does not exists, create it
                os.makedirs(os.path.dirname(cfg.save_report), exist_ok=True)
                json.dump(report, open(cfg.save_report, "w"))
        else:
            print("No task pool specified. Skipping evaluation.")


@hydra.main(
    config_path=os.path.join(
        importlib.import_module("fusion_bench").__path__[0], "../config"
    ),
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
    if cfg.use_lightning:
        program = LightningProgram(cfg)
        program.run_model_fusion()
    else:
        run_model_fusion(cfg)


if __name__ == "__main__":
    main()
