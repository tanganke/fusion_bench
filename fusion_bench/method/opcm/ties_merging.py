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
from fusion_bench.method.ties_merging.ties_merging_utils import (
    state_dict_to_vector,
    ties_merging,
    vector_to_state_dict,
)
from fusion_bench.mixins import LightningFabricMixin, SimpleProfilerMixin
from fusion_bench.taskpool import CLIPVisionModelTaskPool
from fusion_bench.utils.json import load_from_json, save_to_json
from fusion_bench.utils.state_dict_arithmetic import state_dict_add, state_dict_sub

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class ContinualTiesMergingForCLIP(
    BaseAlgorithm,
    LightningFabricMixin,
    SimpleProfilerMixin,
):
    def __init__(
        self,
        scaling_factor: float,
        threshold: float,
        remove_keys: Optional[List[str]] = None,
        merge_func: Literal["sum", "mean", "max"] = "sum",
        shuffle_order: bool = True,
        seed: Optional[int] = None,
        save_on_every_step: bool = True,
        evaluate_on_every_step: bool = False,
        **kwargs,
    ):
        """
        Continual Model Merging via Ties-Merging.

        Args:
            scaling_factor (float): the scaling factor to use.
            shuffle_order (bool): whether to shuffle the order of the models.
            seed (Optional[int]): the seed to use.
            save_on_every_step (bool): whether to save the merged model on every step.
            evaluate_on_every_step (bool): whether to evaluate the merged model on every step.
        """
        self.scaling_factor = scaling_factor
        self.threshold = threshold
        self.remove_keys = remove_keys if remove_keys is not None else []
        self.merge_func = merge_func
        self.shuffle_order = shuffle_order
        self.seed = seed
        self.save_on_every_step = save_on_every_step
        self.evaluate_on_every_step = evaluate_on_every_step
        super().__init__(**kwargs)

    @torch.no_grad()
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
        with self.profile("loading model"):
            pretrained_model = modelpool.load_pretrained_model()
        merged_model = deepcopy(pretrained_model)

        for model_idx, model_name in tqdm(
            enumerate(model_names), desc="Processing models"
        ):
            with self.profile("loading model"):
                task_model = modelpool.load_model(model_name)

            with self.profile("merging model"):
                task_vector = state_dict_sub(
                    task_model.state_dict(),
                    pretrained_model.state_dict(),
                )
                if model_idx == 0:
                    # if is the first model, the merged task vector is equal to the task vector
                    ties_merging_state_dict = task_vector
                else:
                    # if is not the first model, we need to merge the task vector with the previous merged task vector
                    merged_tv = state_dict_sub(
                        merged_model.state_dict(),
                        pretrained_model.state_dict(),
                    )
                    tv_flat_checks = torch.vstack(
                        [
                            state_dict_to_vector(
                                merged_tv, remove_keys=self.remove_keys
                            ),
                            state_dict_to_vector(
                                task_vector, remove_keys=self.remove_keys
                            ),
                        ]
                    )
                    # perform the TIES merging
                    ties_merging_tv = ties_merging(
                        tv_flat_checks,
                        reset_thresh=self.threshold,
                        merge_func=self.merge_func,
                    )
                    # convert the merged task vector back to a state dict
                    ties_merging_state_dict = vector_to_state_dict(
                        ties_merging_tv,
                        merged_model.state_dict(),
                        remove_keys=self.remove_keys,
                    )

                for param_name, param in task_model.named_parameters():
                    if not param.requires_grad:
                        continue

                    merged_param = merged_model.get_parameter(param_name)
                    new_param = (
                        merged_param
                        + self.scaling_factor * ties_merging_state_dict[param_name]
                    )
                    merged_model.get_parameter(param_name).data = new_param

            if self.save_on_every_step:
                with self.profile("saving model"):
                    self.save_merged_model(merged_model, model_idx)

            if self.evaluate_on_every_step:
                with self.profile("evaluating model"):
                    self.taskpool._is_setup = False
                    self.taskpool._test_datasets = DictConfig(
                        {
                            n: self._test_datasets[n]
                            for n in model_names[: model_idx + 1]
                        }
                    )
                    report = self.taskpool.evaluate(deepcopy(merged_model))
                    save_to_json(
                        report, Path(self.log_dir) / f"report_{model_idx}.json"
                    )

        self.print_profile_summary()
        return merged_model

    def save_merged_model(self, merged_model: CLIPVisionModel, step: int):
        os.makedirs(Path(self.log_dir) / "checkpoints", exist_ok=True)
        torch.save(
            merged_model.state_dict(),
            Path(self.log_dir) / "checkpoints" / f"model_{step}.pth",
        )
