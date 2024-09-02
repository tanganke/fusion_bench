import itertools
import logging
import os
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, cast

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
from transformers.models.clip.modeling_clip import CLIPEncoder, CLIPEncoderLayer
from typing_extensions import override

from fusion_bench.method.adamerging.entropy_loss import entropy_loss
from fusion_bench.method.base_algorithm import ModelFusionAlgorithm
from fusion_bench.method.task_arithmetic import task_arithmetic_merge
from fusion_bench.mixins.clip_classification import CLIPClassificationMixin
from fusion_bench.mixins.lightning_fabric import LightningFabricMixin
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import ModelPool
from fusion_bench.modelpool.huggingface_clip_vision import HuggingFaceClipVisionPool
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.models.separate_io import *
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.utils import timeit_context
from fusion_bench.utils.data import InfiniteDataLoader
from fusion_bench.utils.parameters import print_parameters

from .module import ParetoWeightEnsemblingModule
from .utils import generate_simplex_grid

log = logging.getLogger(__name__)


class PWEMoEAlgorithmForCLIP(
    ModelFusionAlgorithm,
    SimpleProfilerMixin,
    CLIPClassificationMixin,
):
    modelpool: HuggingFaceClipVisionPool = None

    @override
    def run(self, modelpool: HuggingFaceClipVisionPool):
        config = self.config
        self.modelpool = modelpool

        model = self.setup_model()
        if config.checkpoint_path is not None:
            model.load_state_dict(
                torch.load(config.checkpoint_path, map_location="cpu")
            )
        else:
            train_loaders = self.setup_train_loaders()
            model = self.train(model, train_loaders)

        if config.eval_grid:
            return map(
                lambda m, r: {
                    "model": ParetoWeightEnsemblingModule.set_preferenece_vector(
                        m,
                        torch.as_tensor(
                            r, device=self.fabric.device, dtype=torch.float32
                        ),
                    ),
                    "preference_vector": r,
                },
                itertools.cycle([model]),
                generate_simplex_grid(config.eval_grid_n, config.eval_grid_m),
            )
        return model

    def load_clip_models(self):
        """
        Loads the pretrained CLIP model and the fine-tuned models for each dataset specified in the configuration.
        """
        # load pretrained and fine-tuned model
        with timeit_context():
            log.info("load models")
            pretrained_model: CLIPVisionModel = self.modelpool.load_model(
                "_pretrained_"
            )
            finetuned_models = {
                model_name: self.modelpool.load_model(model_name)
                for model_name in self.modelpool.model_names
            }

        log.info("pretrained model statistics:")
        print_parameters(pretrained_model)
        return pretrained_model, finetuned_models

    def setup_model(self):
        config = self.config
        pretrained_model, finetuned_models = self.load_clip_models()
        self.setup_zero_shot_classification_head()

        with timeit_context("Building PWEMoE model"):
            model = deepcopy(pretrained_model)

            # merge the remaining layers using task arithmetic
            if config.init_lambda != 0:
                task_arithmetic_merge(
                    model,
                    finetuned_models.values(),
                    scaling_factor=config.init_lambda,
                    inplace=True,
                )
            # fix all parameters
            model.requires_grad_(False)

            num_layers = len(model.vision_model.encoder.layers)
            get_layer = lambda m, i: cast(
                CLIPEncoderLayer, m.vision_model.encoder.layers[i]
            )
            for layer_idx in tqdm(range(num_layers)):
                if config.upscale_mlp:
                    # upscale the mlp layer
                    get_layer(model, layer_idx).mlp = ParetoWeightEnsemblingModule(
                        base_model=get_layer(pretrained_model, layer_idx).mlp,
                        expert_models=[
                            get_layer(m, layer_idx).mlp
                            for m in finetuned_models.values()
                        ],
                        init_lambda=config.init_lambda,
                        fix_base_model_and_experts=True,
                        router_hidden_layers=config.router_hidden_layers,
                    )

                if config.upscale_attn:
                    # upscale the Attention layer
                    get_layer(model, layer_idx).self_attn = (
                        ParetoWeightEnsemblingModule(
                            base_model=get_layer(pretrained_model, layer_idx).self_attn,
                            expert_models=[
                                get_layer(m, layer_idx).self_attn
                                for m in finetuned_models.values()
                            ],
                            init_lambda=config.init_lambda,
                            fix_base_model_and_experts=True,
                            router_hidden_layers=config.router_hidden_layers,
                        )
                    )

            print("model statistics after upscaling:")
            print_parameters(model)
            return model

    def setup_train_loaders(self):
        """
        Loads the datasets specified in the configuration.
        """
        config = self.config
        train_datasets = {
            dataset_name: self.modelpool.get_train_dataset(
                dataset_name, self.clip_processor
            )
            for dataset_name in self.modelpool.model_names
        }
        train_loaders = {
            dataset_name: DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=True,
            )
            for dataset_name, dataset in train_datasets.items()
        }
        train_loaders = {
            dataset_name: self.fabric.setup_dataloaders(loader)
            for dataset_name, loader in train_loaders.items()
        }
        return train_loaders

    def train(self, model: nn.Module, train_loaders: Dict[str, DataLoader]):
        config = self.config

        # save the configuration
        self.log_hyperparams(config, filename="method_config.yaml")

        # setup the model
        num_objectives = len(self.modelpool.model_names)
        model = model

        # setup data loaders
        train_loaders = {
            name: InfiniteDataLoader(loader) for name, loader in train_loaders.items()
        }

        # set up the optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.lr,
        )
        model, optimizer = self.fabric.setup(model, optimizer)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=config.num_steps, eta_min=config.lr * 0.1
        )

        model.train()
        device = self.fabric.device
        for step_idx in tqdm(
            range(1, 1 + config.num_steps), "training", dynamic_ncols=True
        ):
            # sample a preference ray
            ray = torch.from_numpy(
                np.random.dirichlet((config.alpha,) * num_objectives, 1)
                .astype(np.float32)
                .flatten()
            ).to(device)
            ParetoWeightEnsemblingModule.set_preferenece_vector(model, ray)

            losses = []
            for dataset_idx, dataset_name in enumerate(train_loaders):
                batch = next(train_loaders[dataset_name])
                images, labels = batch

                logits = self.compute_logits(model, images, dataset_name)
                _loss = F.cross_entropy(logits, labels)
                losses.append(_loss)

            loss = self.compute_loss(model, ray, losses)

            optimizer.zero_grad()
            self.fabric.backward(loss)
            optimizer.step()

            lr_scheduler.step()

            self.fabric.log("train/loss", loss.item(), step=step_idx)

            if step_idx % config.save_interval == 0:
                (Path(self.log_dir) / "checkpoints").mkdir(exist_ok=True, parents=True)
                save_path = (
                    Path(self.log_dir) / "checkpoints" / f"model_step={step_idx}.pt"
                )
                torch.save(model.state_dict(), save_path)

        return model

    @abstractmethod
    def compute_loss(
        self, model: nn.Module, ray: Tensor, losses: List[Tensor]
    ) -> Tensor:
        """
        Computes the overall losses using the given preference ray.

        Args:
            model (nn.Module): The model being trained.
            ray (Tensor): A tensor representing the preference ray, which contains the weights for each objective.
            losses (List[Tensor]): A list of loss values for each objective.
        """
        pass


class PWEMoELinearScalarizationForCLIP(PWEMoEAlgorithmForCLIP):
    def compute_loss(self, model, ray, losses):
        loss = 0
        for r, l in zip(ray, losses):
            loss += r * l
        return loss


class PWEMoExactParetoOptimalForCLIP(PWEMoEAlgorithmForCLIP):
    def compute_loss(self, model: nn.Module, ray: Tensor, losses: Tuple[Tensor]):
        from phn.solvers import EPOSolver

        if self.epo_solver is None:
            num_objectives = len(self.finetuned_models)
            self.epo_solver = EPOSolver(n_tasks=num_objectives, n_params=None)
        epo_solver = self.epo_solver

        losses = torch.stack(losses)
        loss = epo_solver.get_weighted_loss(
            losses,
            ray,
            tuple(filter(lambda p: p.requires_grad, model.parameters())),
        )
        return loss
