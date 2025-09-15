import json
import os
from typing import Any, Dict, Iterable, List, Optional, Union

import lightning as L
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm.auto import tqdm

from fusion_bench import (
    BaseAlgorithm,
    BaseHydraProgram,
    BaseModelPool,
    BaseTaskPool,
    RuntimeConstants,
    auto_register_config,
    get_rankzero_logger,
    import_object,
    instantiate,
    timeit_context,
)
from fusion_bench.utils.json import print_json
from fusion_bench.utils.rich_utils import print_bordered, print_config_tree

log = get_rankzero_logger(__name__)


@auto_register_config
class ModelFusionProgram(BaseHydraProgram):
    method: BaseAlgorithm
    modelpool: BaseModelPool
    taskpool: Optional[BaseTaskPool] = None

    _config_mapping = BaseHydraProgram._config_mapping | {
        "_method": "method",
        "_modelpool": "modelpool",
        "_taskpool": "taskpool",
        "fast_dev_run": "fast_dev_run",
        "seed": "seed",
        "path": "path",
    }

    def __init__(
        self,
        method: DictConfig,
        modelpool: DictConfig,
        taskpool: Optional[DictConfig] = None,
        *,
        print_config: bool = True,
        dry_run: bool = False,
        report_save_path: Optional[str] = None,
        merged_model_save_path: Optional[str] = None,
        merged_model_save_kwargs: Optional[DictConfig] = None,
        fast_dev_run: bool = False,
        seed: Optional[int] = None,
        print_function_call: bool = True,
        path: DictConfig = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._method = method
        self._modelpool = modelpool
        self._taskpool = taskpool
        self.report_save_path = report_save_path
        self.merged_model_save_path = merged_model_save_path
        self.merged_model_save_kwargs = merged_model_save_kwargs
        self.fast_dev_run = fast_dev_run
        self.seed = seed
        self.path = path
        RuntimeConstants.debug = fast_dev_run
        RuntimeConstants.print_function_call = print_function_call
        if path is not None:
            RuntimeConstants.cache_dir = path.get("cache_dir", None)

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
        R"""
        Instantiates and sets up an object based on the provided configuration.

        This method performs the following steps:
        1. Checks if the configuration dictionary contains the key "_target_".
        2. If "_target_" is not found (for v0.1.x), attempts to instantiate the object using a compatible load function if provided.
           - Logs a warning if "_target_" is missing.
           - If `compat_load_fn` is provided, imports the function and uses it to instantiate the object.
           - If `compat_load_fn` is not provided, raises a ValueError.
        3. If "_target_" is found (for v.0.2.0 and above), attempts to import and instantiate the object using the `instantiate` function.
           - Ensures the target can be imported.
           - Uses the `instantiate` function with `_recursive_` set based on the configuration.
        4. Sets the `_program` attribute of the instantiated object to `self` if the object has this attribute.
        5. Returns the instantiated and set up object.
        """
        if "_target_" not in config:
            log.warning(
                "No '_target_' key found in config. Attempting to instantiate the object in a compatible way."
            )
            if compat_load_fn is not None:
                compat_load_fn = import_object(compat_load_fn)
                if rank_zero_only.rank == 0:
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
        return obj

    def save_merged_model(self, merged_model):
        """
        Saves the merged model to the specified path.
        """
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
        self,
        taskpool: BaseTaskPool,
        merged_model: Union[nn.Module, Dict, Iterable],
        *args: Any,
        **kwargs: Any,
    ) -> Union[Dict, List, Any]:
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
            *args: Additional positional arguments to be passed to the `evaluate` method of the taskpool.
            **kwargs: Additional keyword arguments to be passed to the `evaluate` method of the taskpool.

        Returns:
            The evaluation report. The type of the report depends on the type of the merged model:
            - If the merged model is an instance of `nn.Module`, the report is a dictionary.
            - If the merged model is a dictionary, the report is a dictionary updated with the remaining dictionary items.
            - If the merged model is an iterable, the report is a list of evaluation reports.
        """
        if isinstance(merged_model, nn.Module):
            report = taskpool.evaluate(merged_model, *args, **kwargs)
            return report
        elif isinstance(merged_model, Dict):
            report = {}
            for key, item in merged_model.items():
                if isinstance(item, nn.Module):
                    report[key] = taskpool.evaluate(item, *args, **kwargs)
                elif key == "models":
                    # for multi-model evaluation
                    report[key] = self.evaluate_merged_model(
                        taskpool, item, *args, **kwargs
                    )
                else:
                    # metadata
                    report[key] = item
            return report
        elif isinstance(merged_model, Iterable):
            return [
                self.evaluate_merged_model(taskpool, m, *args, **kwargs)
                for m in tqdm(merged_model, desc="Evaluating models")
            ]
        else:
            raise ValueError(f"Invalid type for merged model: {type(merged_model)}")

    def run(self):
        """
        Executes the model fusion program.
        """
        if self.seed is not None:
            L.seed_everything(self.seed)

        log.info("Running the model fusion program.")
        # setup the modelpool, method, and taskpool
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
        if self._taskpool is not None:
            log.info("loading task pool")
            self.taskpool = self._instantiate_and_setup(
                self._taskpool,
                compat_load_fn="fusion_bench.compat.taskpool.load_taskpool_from_config",
            )

        self.method.on_run_start()
        merged_model = self.method.run(self.modelpool)
        self.method.on_run_end()

        if merged_model is None:
            log.info(
                "No merged model returned by the method. Skipping saving and evaluation."
            )
        else:
            self.save_merged_model(merged_model)
            if self.taskpool is not None:
                report = self.evaluate_merged_model(self.taskpool, merged_model)
                try:
                    if rank_zero_only.rank == 0:
                        print_json(report, print_type=False)
                except Exception as e:
                    log.warning(f"Failed to pretty print the report: {e}")
                    log.info(report)
                if self.report_save_path is not None:
                    # save report (Dict) to a file
                    # if the directory of `save_report` does not exists, create it
                    if (
                        "{log_dir}" in self.report_save_path
                        and self.path.log_dir is not None
                    ):
                        self.report_save_path = self.report_save_path.format(
                            log_dir=self.path.log_dir
                        )
                    os.makedirs(os.path.dirname(self.report_save_path), exist_ok=True)
                    json.dump(report, open(self.report_save_path, "w"))
            else:
                log.info("No task pool specified. Skipping evaluation.")
