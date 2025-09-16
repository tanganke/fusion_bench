import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Iterator, Optional, Union, override

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fusion_bench import (
    BaseAlgorithm,
    LightningFabricMixin,
    auto_register_config,
    get_rankzero_logger,
    instantiate,
)
from fusion_bench.constants import RuntimeConstants
from fusion_bench.dataset import CLIPDataset
from fusion_bench.modelpool import ResNetForImageClassificationPool
from fusion_bench.models.wrappers.layer_wise_fusion import LayerWiseMergedModel
from fusion_bench.models.wrappers.task_wise_fusion import TaskWiseMergedModel
from fusion_bench.utils import load_tensor_from_file
from fusion_bench.utils.data import InfiniteDataLoader

from .entropy_loss import entropy_loss
from .utils import construct_layer_wise_merged_model, construct_task_wise_merged_model

if TYPE_CHECKING:
    from transformers import ResNetForImageClassification, ResNetModel

log = get_rankzero_logger(__name__)


@auto_register_config
class _ResNetAdaMergingBase(
    ABC,
    LightningFabricMixin,
    BaseAlgorithm,
):
    classification_heads: Dict[str, nn.Module]
    shuffled_test_loader_iters: Dict[str, Iterator]

    def __init__(
        self,
        max_steps: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        dataloader_kwargs: DictConfig,
        init_values: Optional[float],
        clamp_weights: bool = False,
        tie_weights: bool = True,
        strict: bool = False,
        resume_weights_path: Union[str, None] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if RuntimeConstants.debug:
            log.info("Debug mode is on, setting max_steps to 10")
            self.max_steps = 10

    @override
    def run(self, modelpool: ResNetForImageClassificationPool):
        self.modelpool = modelpool

        # setup models
        wrapped_model = self.setup_wrapped_model(modelpool)

        # if max_steps <= 0, skip training and return the merged model directly
        # this can be used to evaluate the merging weights loaded from `resume_weights_path`
        if self.max_steps <= 0:
            # skip_training
            return wrapped_model.merge_and_unload()

        # setup dataloaders
        self.setup_dataloaders()

        # configure optimizer and lr_scheduler
        optimizer = instantiate(self.optimizer, params=[wrapped_model.merge_weight])
        if self.lr_scheduler is not None:
            lr_scheduler = instantiate(self.lr_scheduler, optimizer=optimizer)
        else:
            lr_scheduler = None

        wrapped_model, optimizer = self.fabric.setup(wrapped_model, optimizer)
        wrapped_model = self.test_time_adaptation(
            wrapped_model, optimizer, lr_scheduler
        )

        # save merging weights
        if self.log_dir is not None:
            self.fabric.save(
                os.path.join(self.log_dir, "checkpoints", "merge_weight.ckpt"),
                {"merge_weight": wrapped_model.merge_weight},
            )

        merged_model = wrapped_model.merge_and_unload()
        if self.log_dir is not None:
            modelpool.save_model(
                merged_model,
                os.path.join(self.log_dir, "checkpoints", "merged_model"),
                algorithm_config=self.config,
                description="Merged ResNet model using AdaMerging (E Yang, 2023).",
            )

        return merged_model

    def test_time_adaptation(
        self,
        wrapped_model: TaskWiseMergedModel,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    ):
        model_names = self.modelpool.model_names
        wrapped_model.train()
        wrapped_model.merge_weights()

        for step_idx in tqdm(
            range(self.max_steps),
            disable=not self.fabric.is_global_zero,
            dynamic_ncols=True,
        ):
            metrics = {"tta/total_loss": 0.0}
            for task in model_names:
                batch = next(self.get_shuffled_test_loader_iter(task))
                logits = self.compute_logits(wrapped_model, batch[0], task)
                loss = entropy_loss(logits)
                metrics[f"tta/{task}_loss"] = loss.item()
                metrics["tta/total_loss"] += loss.item()
                self.fabric.backward(loss, retain_graph=True)

            optimizer.step()
            optimizer.zero_grad()
            wrapped_model.merge_weights()  # merge weights for the next step
            if lr_scheduler is not None:
                lr_scheduler.step()

            self.fabric.log_dict(metrics=metrics, step=step_idx)

        return wrapped_model

    def compute_logits(
        self, module: Union["ResNetModel", nn.Module], images: torch.Tensor, task: str
    ) -> torch.Tensor:
        if self.modelpool.type == "transformers":
            outputs = module(images, return_dict=True)
            pooled_output = outputs.pooler_output
            logits = self.classification_heads[task](pooled_output)
            return logits
        else:
            raise NotImplementedError(
                f"Model type {self.modelpool.type} is not supported."
            )

    def setup_dataloaders(self):
        dataloader_kwargs = dict(self.dataloader_kwargs)
        dataloader_kwargs["shuffle"] = True  # ensure shuffling for TTA
        processor = self.modelpool.load_processor()
        for task in self.modelpool.test_dataset_names:
            test_dataset = self.modelpool.load_test_dataset(task)
            test_dataset = CLIPDataset(test_dataset, processor=processor)
            test_loader = DataLoader(test_dataset, **dataloader_kwargs)
            self.shuffled_test_loader_iters[task] = iter(
                InfiniteDataLoader(test_loader)
            )

    def get_shuffled_test_loader_iter(self, task: str):
        return self.shuffled_test_loader_iters[task]

    @abstractmethod
    def setup_wrapped_model(
        self, modelpool: ResNetForImageClassificationPool
    ) -> Union[TaskWiseMergedModel, LayerWiseMergedModel]:
        """
        Setup the wrapped merged model.

        Args:
            modelpool (ResNetForImageClassificationPool): The model pool containing pretrained and finetuned models.

        Returns:
            Union[TaskWiseMergedModel, LayerWiseMergedModel] : The wrapped merged model.
        """
        pass


class ResNetTaskWiseAdamerging(_ResNetAdaMergingBase):
    @torch.no_grad()
    def setup_wrapped_model(self, modelpool: ResNetForImageClassificationPool):
        pretrained_model = modelpool.load_pretrained_model()
        finetuned_models = dict(modelpool.named_models())

        if modelpool.type == "transformers":
            pretrained_model: "ResNetForImageClassification"
            finetuned_models: Dict[str, "ResNetForImageClassification"]
            for model_name in finetuned_models:
                self.classification_heads[model_name] = finetuned_models[
                    model_name
                ].classifier
                # fix the classification head during merging and move to device
                self.classification_heads[model_name].requires_grad_(False)
            pretrained_backbone: "ResNetModel" = pretrained_model.resnet
            finetuned_backbones = [
                finetuned_models[model_name].resnet for model_name in finetuned_models
            ]
        else:
            raise NotImplementedError(f"Model type {modelpool.type} is not supported.")

        wrapped_model = construct_task_wise_merged_model(
            pretrained_model=pretrained_backbone,
            finetuned_models=finetuned_backbones,
            clamp_weights=self.clamp_weights,
            tie_weights=self.tie_weights,
            strict=self.strict,
        )

        if self.init_values is not None:
            log.info(f"Initializing merging weights to {self.init_values}")
            wrapped_model.merge_weight.data.fill_(self.init_values)

        # load merging weights if provided
        if self.resume_weights_path is not None:
            merging_weights = load_tensor_from_file(
                self.resume_weights_path, device="cpu"
            )
            log.info(f"Loaded merging weights from {self.resume_weights_path}")
            assert merging_weights.shape == wrapped_model.merge_weight.shape, (
                f"Merging weights shape {merging_weights.shape} does not match "
                f"model's merge_weight shape {wrapped_model.merge_weight.shape}."
            )
            wrapped_model.merge_weight.data = merging_weights
        return wrapped_model


class ResNetLayerWiseAdamerging(_ResNetAdaMergingBase):
    @torch.no_grad()
    def setup_wrapped_model(self, modelpool: ResNetForImageClassificationPool):
        pretrained_model = modelpool.load_pretrained_model()
        finetuned_models = dict(modelpool.named_models())

        if modelpool.type == "transformers":
            pretrained_model: "ResNetForImageClassification"
            finetuned_models: Dict[str, "ResNetForImageClassification"]
            for model_name in finetuned_models:
                self.classification_heads[model_name] = finetuned_models[
                    model_name
                ].classifier
                # fix the classification head during merging and move to device
                self.classification_heads[model_name].requires_grad_(False)
            pretrained_backbone: "ResNetModel" = pretrained_model.resnet
            finetuned_backbones = [
                finetuned_models[model_name].resnet for model_name in finetuned_models
            ]
        else:
            raise NotImplementedError(f"Model type {modelpool.type} is not supported.")

        wrapped_model = construct_layer_wise_merged_model(
            pretrained_model=pretrained_backbone,
            finetuned_models=finetuned_backbones,
            clamp_weights=self.clamp_weights,
            tie_weights=self.tie_weights,
            strict=self.strict,
        )

        if self.init_values is not None:
            log.info(f"Initializing merging weights to {self.init_values}")
            wrapped_model.merge_weight.data.fill_(self.init_values)

        # load merging weights if provided
        if self.resume_weights_path is not None:
            merging_weights = load_tensor_from_file(
                self.resume_weights_path, device="cpu"
            )
            log.info(f"Loaded merging weights from {self.resume_weights_path}")
            assert merging_weights.shape == wrapped_model.merge_weight.shape, (
                f"Merging weights shape {merging_weights.shape} does not match "
                f"model's merge_weight shape {wrapped_model.merge_weight.shape}."
            )
            wrapped_model.merge_weight.data = merging_weights
        return wrapped_model
