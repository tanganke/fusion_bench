import importlib
import importlib.resources
import json
import logging
import os
from typing import Dict, Iterable, Optional, Union

import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm.auto import tqdm

from fusion_bench.method import BaseModelFusionAlgorithm, load_algorithm_from_config
from fusion_bench.mixins import LightningFabricMixin, YAMLSerializationMixin
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.scripts import BaseHydraProgram
from fusion_bench.taskpool import BaseTaskPool
from fusion_bench.utils import timeit_context
from fusion_bench.utils.rich_utils import print_config_tree

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
        "_usage_": "_usage_",
        "_version_": "_version_",
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
        _usage_: Optional[str] = None,
        _version_: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self._method = method
        self._modelpool = modelpool
        self._taskpool = taskpool
        self._fabric = fabric
        self._fabric_logger = fabric_logger
        self._usage_ = _usage_
        self._version_ = _version_
        self.report_save_path = report_save_path
        self.merged_model_save_path = merged_model_save_path
        self.merged_model_save_kwargs = merged_model_save_kwargs

        if print_config:
            print_config_tree(
                self.config,
                print_order=["method", "modelpool", "taskpool"],
            )
        if dry_run:
            log.info("The program is running in dry-run mode. Exiting.")
            return

    def _instantiate_and_setup(self, config: DictConfig):
        assert "_target_" in config, f"Missing '_target_' in config: {config}"
        obj = instantiate(config)
        if hasattr(obj, "_program"):
            obj._program = self
        if hasattr(obj, "_fabric") and self.fabric is not None:
            obj._fabric = self.fabric
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
        self.modelpool = self._instantiate_and_setup(self._modelpool)
        self.method = self._instantiate_and_setup(self._method)
        merged_model = self.method.run(self.modelpool)
        self.save_merged_model(merged_model)

        if self._taskpool is not None:
            self.taskpool = self._instantiate_and_setup(self._taskpool)
            report = self.evaluate_merged_model(self.taskpool, merged_model)
            print(report)
            if self.report_save_path is not None:
                # save report (Dict) to a file
                # if the directory of `save_report` does not exists, create it
                os.makedirs(os.path.dirname(self.report_save_path), exist_ok=True)
                json.dump(report, open(self.report_save_path, "w"))
        else:
            print("No task pool specified. Skipping evaluation.")
