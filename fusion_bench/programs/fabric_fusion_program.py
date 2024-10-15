import importlib
import importlib.resources
import json
import logging
import os
from typing import Callable, Dict, Iterable, Optional, Union

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm.auto import tqdm

import fusion_bench.utils.instantiate
from fusion_bench.method import BaseModelFusionAlgorithm
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.programs import BaseHydraProgram
from fusion_bench.taskpool import BaseTaskPool
from fusion_bench.utils import import_object, instantiate, timeit_context
from fusion_bench.utils.hydra_utils import get_hydra_output_dir
from fusion_bench.utils.rich_utils import print_config_tree, print_bordered
from fusion_bench.utils.json import print_json

log = logging.getLogger(__name__)


class FabricModelFusionProgram(
    LightningFabricMixin,
    BaseHydraProgram,
):
    method: BaseModelFusionAlgorithm
    modelpool: BaseModelPool
    taskpool: Optional[BaseTaskPool] = None

    _config_mapping = BaseHydraProgram._config_mapping | {
        "_method": "method",
        "_modelpool": "modelpool",
        "_taskpool": "taskpool",
        "_fabric": "fabric",
        "_fabric_logger": "fabric_logger",
        "fast_dev_run": "fast_dev_run",
        "seed": "seed",
    }

    def __init__(
        self,
        method: DictConfig,
        modelpool: DictConfig,
        taskpool: Optional[DictConfig] = None,
        *,
        fabric: Optional[DictConfig] = None,
        fabric_logger: Optional[DictConfig] = None,
        print_config: bool = True,
        dry_run: bool = False,
        report_save_path: Optional[str] = None,
        merged_model_save_path: Optional[str] = None,
        merged_model_save_kwargs: Optional[DictConfig] = None,
        fast_dev_run: bool = False,
        seed: Optional[int] = None,
        print_function_call: bool = True,
        **kwargs,
    ):
        self._method = method
        self._modelpool = modelpool
        self._taskpool = taskpool
        self._fabric = fabric
        self._fabric_logger = fabric_logger
        self.report_save_path = report_save_path
        self.merged_model_save_path = merged_model_save_path
        self.merged_model_save_kwargs = merged_model_save_kwargs
        self.fast_dev_run = fast_dev_run
        self.seed = seed
        super().__init__(**kwargs)
        fusion_bench.utils.instantiate.PRINT_FUNCTION_CALL = print_function_call

        if print_config:
            print_config_tree(
                self.config,
                print_order=["method", "modelpool", "taskpool"],
            )
        if dry_run:
            log.info("The program is running in dry-run mode. Exiting.")
            exit(0)

    def _instantiate_and_setup(
        self, config: DictConfig, compat_load_fn: Optional[str] = None
    ):
        if "_target_" not in config:
            log.warning(
                "No '_target_' key found in config. Attempting to instantiate the object in a compatible way."
            )
            if compat_load_fn is not None:
                compat_load_fn = import_object(compat_load_fn)
                print_bordered(
                    OmegaConf.to_yaml(config),
                    title="instantiate compat object",
                    style="magenta",
                    code_style="yaml",
                )
                obj = compat_load_fn(config)
            else:
                raise ValueError(
                    "No load function provided. Please provide a load function to instantiate the object."
                )
        else:
            # try to import the object from the target
            # this checks if the target is valid and can be imported
            import_object(config._target_)
            obj = instantiate(
                config,
                _recursive_=config.get("_recursive_", False),
            )
        if hasattr(obj, "_program"):
            obj._program = self
        if hasattr(obj, "_fabric_instance") and self.fabric is not None:
            obj._fabric_instance = self.fabric
        return obj

    def save_merged_model(self, merged_model):
        if self.merged_model_save_path is not None:
            # path to save the merged model, use "{log_dir}" to refer to the logger directory
            save_path: str = self.merged_model_save_path
            if "{log_dir}" in save_path and self.log_dir is not None:
                save_path = save_path.format(log_dir=self.log_dir)

            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # save the merged model
            if self.merged_model_save_kwargs is not None:
                merged_model_save_kwargs = self.merged_model_save_kwargs
            else:
                merged_model_save_kwargs = {}
            with timeit_context(f"Saving the merged model to {save_path}"):
                self.modelpool.save_model(
                    merged_model,
                    save_path,
                    **merged_model_save_kwargs,
                )
        else:
            print("No save path specified for the merged model. Skipping saving.")

    def evaluate_merged_model(
        self, taskpool: BaseTaskPool, merged_model: Union[nn.Module, Dict, Iterable]
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
            merged_model: The merged model to be evaluated. It can be an instance of `nn.Module`, a dictionary, or an iterable.

        Returns:
            The evaluation report. The type of the report depends on the type of the merged model:
            - If the merged model is an instance of `nn.Module`, the report is a dictionary.
            - If the merged model is a dictionary, the report is a dictionary updated with the remaining dictionary items.
            - If the merged model is an iterable, the report is a list of evaluation reports.
        """
        if isinstance(merged_model, nn.Module):
            report = taskpool.evaluate(merged_model)
            return report
        elif isinstance(merged_model, Dict):
            model = merged_model.pop("model")
            report: dict = taskpool.evaluate(model)
            report.update(merged_model)
            print(report)
            return report
        elif isinstance(merged_model, Iterable):
            return [
                self.evaluate_merged_model(taskpool, m)
                for m in tqdm(merged_model, desc="Evaluating models")
            ]
        else:
            raise ValueError(f"Invalid type for merged model: {type(merged_model)}")

    def run(self):
        if self.seed is not None:
            L.seed_everything(self.seed)

        self._link_hydra_output()

        log.info("Running the model fusion program.")
        log.info("loading model pool")
        self.modelpool = self._instantiate_and_setup(
            self._modelpool,
            compat_load_fn="fusion_bench.compat.modelpool.load_modelpool_from_config",
        )
        log.info("loading method")
        self.method = self._instantiate_and_setup(
            self._method,
            compat_load_fn="fusion_bench.compat.method.load_algorithm_from_config",
        )
        merged_model = self.method.run(self.modelpool)
        self.save_merged_model(merged_model)

        if self._taskpool is not None:
            log.info("loading task pool")
            self.taskpool = self._instantiate_and_setup(
                self._taskpool,
                compat_load_fn="fusion_bench.compat.taskpool.load_taskpool_from_config",
            )
            report = self.evaluate_merged_model(self.taskpool, merged_model)
            print_json(report, print_type=False)
            if self.report_save_path is not None:
                # save report (Dict) to a file
                # if the directory of `save_report` does not exists, create it
                os.makedirs(os.path.dirname(self.report_save_path), exist_ok=True)
                json.dump(report, open(self.report_save_path, "w"))
        else:
            print("No task pool specified. Skipping evaluation.")

    def _link_hydra_output(self):
        if self.log_dir is not None:
            # make symlink to the hydra output directory
            hydra_output_dir = get_hydra_output_dir()
            if hydra_output_dir is not None:
                os.makedirs(self.log_dir, exist_ok=True)
                os.symlink(
                    hydra_output_dir,
                    os.path.join(
                        self.log_dir,
                        "hydra_output_" + os.path.basename(hydra_output_dir),
                    ),
                    target_is_directory=True,
                )
