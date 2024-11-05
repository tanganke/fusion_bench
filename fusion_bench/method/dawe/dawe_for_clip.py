# NOTE: Working in progress.
import logging
from pathlib import Path
from typing import Any, Literal, Optional, Union  # noqa: F401

import torch
from omegaconf import DictConfig
from PIL.Image import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoFeatureExtractor,
    CLIPProcessor,
    PreTrainedModel,
    ResNetForImageClassification,
)

from fusion_bench.dataset.clip_dataset import CLIPDataset
from fusion_bench.method import BaseAlgorithm
from fusion_bench.method.adamerging.entropy_loss import entropy_loss
from fusion_bench.mixins import CLIPClassificationMixin
from fusion_bench.modelpool import CLIPVisionModelPool
from fusion_bench.utils import timeit_context
from fusion_bench.utils.data import InfiniteDataLoader
from fusion_bench.utils.instantiate import instantiate

from .warppers.dawe_model import DataAdaptiveWeightEnsemblingCLIPVisionModel

log = logging.getLogger(__name__)


def convert_to_rgb(image: Image | list[Image]) -> Image | list[Image]:
    if isinstance(image, (list, tuple)):
        return [convert_to_rgb(img) for img in image]
    else:
        return image.convert("RGB")


def load_resnet_processor(pretrained_model_name_or_path: str):
    """
    Load a ResNet processor for image preprocessing.

    Args:
        pretrained_model_name_or_path (str): The path or name of the pretrained ResNet model.

    Returns:
        function: A function that processes images using the ResNet processor.
    """
    processor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path)
    return lambda img: processor(
        images=convert_to_rgb(img), return_tensors="pt", do_rescale=False
    ).pixel_values


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained_model_name_or_path):
        super().__init__()
        self.model = ResNetForImageClassification.from_pretrained(
            pretrained_model_name_or_path
        )
        self.model.classifier = nn.Flatten(1, -1)
        self.config = self.model.config

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        return outputs.logits


def load_resnet_feature_extractor(pretrained_model_name_or_path: str):
    model = ResNetFeatureExtractor(pretrained_model_name_or_path)
    return model


def raw_image_collate_fn(batch):
    images, labels = tuple(zip(*batch))
    labels = torch.as_tensor(labels)
    return images, labels


class DataAdaptiveWeightEnsemblingForCLIP(
    BaseAlgorithm,
    CLIPClassificationMixin,
):
    modelpool: CLIPVisionModelPool
    _processor: CLIPProcessor

    def __init__(
        self,
        # merge options
        merge_mode: Literal["task_wise", "layer_wise"],
        init_lambda: float,
        batch_reduce: bool,
        eval_batch_reduce: bool,
        # model options
        dict_processor: DictConfig,
        dict_feature_extractor: DictConfig,
        hidden_size: Optional[int],
        gate_hidden_layers: int,
        task_vector_dtype: Optional[str | torch.dtype],
        task_vector_sparsity: float,
        # training & logging args
        max_steps: int,
        save_interval: int,
        learning_rate: float = 1e-5,
        skip_training: bool = False,
        resume_checkpoint_path: Optional[str] = None,
        # dataloader args
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = True,
        **kwargs,
    ):
        # merge options
        self.merge_mode = merge_mode
        self.init_lambda = init_lambda
        self.batch_reduce = batch_reduce
        self.eval_batch_reduce = eval_batch_reduce
        # model options
        self._dict_processor = dict_processor
        self._dict_feature_extractor = dict_feature_extractor
        self.hidden_size = hidden_size
        self.gate_hidden_layers = gate_hidden_layers
        self.task_vector_dtype = task_vector_dtype
        self.task_vector_sparsity = task_vector_sparsity
        # training & logging args
        self.max_steps = max_steps
        self.save_interval = save_interval
        self.learning_rate = learning_rate
        self.skip_training = skip_training
        self.resume_checkpoint_path = resume_checkpoint_path
        # dataloader args
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        super().__init__(**kwargs)

    def load_models(self):
        modelpool = self.modelpool

        dict_processor = instantiate(self._dict_processor)
        clip_processor = modelpool.load_processor()

        dict_feature_extractor: Union[PreTrainedModel, nn.Module] = instantiate(
            self._dict_feature_extractor
        )
        if self.hidden_size is None:
            # try to infer hidden size from feature extractor model
            self.hidden_size = dict_feature_extractor.config.hidden_sizes[-1]

        # initialize classification head
        self.setup_zero_shot_classification_head(
            clip_processor=clip_processor,
            task_names=modelpool.model_names,
        )
        model = DataAdaptiveWeightEnsemblingCLIPVisionModel(
            merge_mode=self.merge_mode,
            hidden_size=self.hidden_size,
            dict_processor=dict_processor,
            model_processor=lambda images: clip_processor(
                images=images, return_tensors="pt"
            ).pixel_values,
            collate_fn=lambda outputs: torch.cat(
                [out.pooler_output for out in outputs], dim=0
            ),
            dict_feature_extractor=dict_feature_extractor,
            base_model=modelpool.load_model("_pretrained_"),
            expert_models=list(modelpool.models()),
            task_vector_dtype=self.task_vector_dtype,
            task_vector_sparsity=self.task_vector_sparsity,
            init_lambda=self.init_lambda,
            gate_hidden_layers=self.gate_hidden_layers,
            batch_reduce=self.batch_reduce,
        )

        if self.resume_checkpoint_path is not None:
            self.fabric.load(self.resume_checkpoint_path, {"model": model})
        return model

    def load_datasets(self):
        modelpool = self.modelpool
        self.test_datasets = {
            task_name: CLIPDataset(
                modelpool.load_test_dataset(task_name),
                processor=None,  # NOTE: processor is not used in CLIPDataset because feature extractor and model may have different processors, so we want to pass the image as is
            )
            for task_name in modelpool.model_names
        }

        # setup dataloaders for test-time adaptation training

        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        self.shuffled_test_loaders = {
            task_name: self.fabric.setup_dataloaders(
                DataLoader(
                    test_dataset,
                    **dataloader_kwargs,
                    collate_fn=raw_image_collate_fn,
                    shuffle=True,
                )
            )
            for task_name, test_dataset in self.test_datasets.items()
        }
        self.shuffled_test_loader_iters = {
            task_name: InfiniteDataLoader(loader)
            for task_name, loader in self.shuffled_test_loaders.items()
        }

    def run(self, modelpool: CLIPVisionModelPool):
        self.modelpool = modelpool
        with timeit_context("Loading models"):
            model = self.load_models()
        with timeit_context("Loading dataloaders"):
            self.load_datasets()

        # run test-time adaptation
        if not self.skip_training:
            model = self.test_time_adaptation_training(modelpool, model)

        if self.eval_batch_reduce is not None:
            model.batch_reduce = self.eval_batch_reduce
        return model

    def test_time_adaptation_training(self, modelpool, model):
        optimizer = torch.optim.Adam(
            [p for p in model.gate.parameters() if p.requires_grad],
            lr=self.learning_rate,
        )
        model, optimizer = self.fabric.setup(model, optimizer)
        model.train()
        for step_idx in tqdm(
            range(self.max_steps),
            desc="TTA Training",
            dynamic_ncols=True,
        ):
            log_metrics = {}
            losses = 0
            for task_idx, task_name in enumerate(modelpool.model_names):
                # labels are used for logging acc, not involved in training
                images, labels = next(self.shuffled_test_loader_iters[task_name])
                logits = self.compute_logits(model, images=images, task=task_name)
                loss = entropy_loss(logits)
                losses += loss
                log_metrics[f"train/{task_name}_loss"] = loss.item()
                log_metrics[f"train/{task_name}_accuracy"] = (
                    logits.argmax(dim=-1).eq(labels).float().mean().item()
                )

            optimizer.zero_grad()
            self.fabric.backward(losses)
            optimizer.step()

            log_metrics["train/loss"] = losses.item()
            self.fabric.log_dict(log_metrics, step=step_idx)

            if (step_idx + 1) % self.save_interval == 0:
                log.info(f"Saving model at step {step_idx}")
                self.fabric.save(
                    Path(self.log_dir) / "checkpoints" / f"model_{step_idx}.pt",
                    {"model": model},
                )

        if (step_idx + 1) % self.save_interval != 0:
            # if the last step was not saved, save it now
            self.fabric.save(
                Path(self.log_dir) / "checkpoints" / f"model_{step_idx}.pt",
                {"model": model},
            )

        return model
