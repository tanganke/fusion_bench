import os
import random
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, cast

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import CLIPVisionModel

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.taskpool import CLIPVisionModelTaskPool
from fusion_bench.utils.json import load_from_json, save_to_json

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class ContinualWeightAverageForCLIP(
    BaseAlgorithm,
    LightningFabricMixin,
):
    def __init__(
        self,
        shuffle_order: bool = True,
        seed: Optional[int] = None,
        save_on_every_step: bool = True,
        evaluate_on_every_step: bool = False,
        **kwargs,
    ):
        """
        Continual Model Merging via Weight Average.

        Args:
            shuffle_order (bool): whether to shuffle the order of the models.
            seed (Optional[int]): the seed to use.
            save_on_every_step (bool): whether to save the merged model on every step.
            evaluate_on_every_step (bool): whether to evaluate the merged model on every step.
        """
        self.shuffle_order = shuffle_order
        self.seed = seed
        self.save_on_every_step = save_on_every_step
        self.evaluate_on_every_step = evaluate_on_every_step
        super().__init__(**kwargs)

    def run(self, modelpool: BaseModelPool):
        if self.seed is not None:
            L.seed_everything(self.seed)

        model_names = modelpool.model_names
        if self.shuffle_order:
            random.shuffle(model_names)

        self.taskpool = cast(CLIPVisionModelTaskPool, self._program.taskpool)
        self._test_datasets = deepcopy(self.taskpool._test_datasets)
        """Configuration for the test datasets"""

        # log the model names
        if self.log_dir is not None:
            save_to_json(model_names, Path(self.log_dir) / "model_names.json")
            tensorboard_summarywriter: "SummaryWriter" = self.tensorboard_summarywriter
            tensorboard_summarywriter.add_text(
                "global/model_names", str(model_names), global_step=0
            )

        # get the average model
        merged_model = modelpool.load_model(model_names[0])

        if self.evaluate_on_every_step:
            self.taskpool._is_setup = False
            self.taskpool._test_datasets = DictConfig(
                {model_names[0]: self._test_datasets[model_names[0]]}
            )
            report = self.taskpool.evaluate(deepcopy(merged_model))
            save_to_json(report, Path(self.log_dir) / "report_0.json")

        if self.save_on_every_step:
            self.save_merged_model(merged_model, 0)

        for model_idx, model_name in tqdm(
            enumerate(model_names[1:]), desc="Processing models"
        ):
            model_idx += 1
            task_model = modelpool.load_model(model_name)

            for param_name, param in task_model.named_parameters():
                if not param.requires_grad:
                    continue

                task_param = param
                merged_param = merged_model.get_parameter(param_name)

                new_param = (merged_param * model_idx + task_param) / (model_idx + 1)
                merged_model.get_parameter(param_name).data = new_param

            if self.save_on_every_step:
                self.save_merged_model(merged_model, model_idx)

            if self.evaluate_on_every_step:
                self.taskpool._is_setup = False
                self.taskpool._test_datasets = DictConfig(
                    {n: self._test_datasets[n] for n in model_names[: model_idx + 1]}
                )
                report = self.taskpool.evaluate(deepcopy(merged_model))
                save_to_json(report, Path(self.log_dir) / f"report_{model_idx}.json")

        return merged_model

    def save_merged_model(self, merged_model: CLIPVisionModel, step: int):
        os.makedirs(Path(self.log_dir) / "checkpoints", exist_ok=True)
        merged_model.save_pretrained(
            Path(self.log_dir) / "checkpoints" / f"merged_model_{step}"
        )
