import logging
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

import lightning.fabric
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from open_clip.model import ResidualAttentionBlock
from torch import Tensor, nn
from tqdm.auto import tqdm

from fusion_bench import BaseAlgorithm
from fusion_bench.dataset.clip_dataset import CLIPDataset
from fusion_bench.method.task_arithmetic import task_arithmetic_merge
from fusion_bench.mixins import OpenCLIPClassificationMixin, SimpleProfilerMixin
from fusion_bench.modelpool import OpenCLIPVisionModelPool
from fusion_bench.models.open_clip import ClassificationHead, ImageEncoder
from fusion_bench.utils import print_parameters, timeit_context
from fusion_bench.utils.data import InfiniteDataLoader

from .module import ParetoWeightEnsemblingModule
from .phn.solvers import EPOSolver
from .utils import generate_simplex_grid

log = logging.getLogger(__name__)


class PWEMoEAlgorithmForOpenCLIP(
    BaseAlgorithm,
    SimpleProfilerMixin,
    OpenCLIPClassificationMixin,
):
    modelpool: OpenCLIPVisionModelPool

    #! === Training & Validation Data ===
    # setup the datasets and loaders by calling `load_datasets`
    train_datasets: Dict[str, CLIPDataset]
    train_loaders: Dict[str, torch.utils.data.DataLoader]
    train_loader_iters: Dict[str, Iterator[Tuple[torch.Tensor, torch.Tensor]]]

    test_datasets: Dict[str, CLIPDataset]
    test_loaders: Dict[str, torch.utils.data.DataLoader]

    def __init__(
        self,
        *,
        #! === Model Architecture Arguments ===
        partial: bool,
        init_lambda: float,
        router_hidden_layers: int,
        checkpoint_path: str,
        #! === Training Arguments ===
        run_train: bool,
        num_steps: int,
        save_interval: int,
        lr: float,
        alpha: float,
        dataloader_kwargs: DictConfig,
        #! === Evaluation Arguments ===
        run_eval: bool,
        num_evaluation_samples: Union[str, int],
        quick_evaluation: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.partial = partial
        self.init_lambda = init_lambda
        self.router_hidden_layers = router_hidden_layers
        self.lr = lr
        self.num_steps = num_steps
        self.save_interval = save_interval
        self.alpha = alpha
        self.checkpoint_path = checkpoint_path
        self._dataloader_kwargs = dataloader_kwargs
        self.run_train = run_train
        self.run_eval = run_eval
        self.num_evaluation_samples = num_evaluation_samples
        self.quick_evaluation = quick_evaluation

    def run(self, modelpool: OpenCLIPVisionModelPool):
        self.modelpool = modelpool

        # setup the MoE model
        model = self.load_model()
        if self.checkpoint_path is not None:
            self.fabric.load(self.checkpoint_path, {"model": model})

        # setup dataloaders
        self.load_datasets()

        if self.run_train:
            model = self.train()
        if self.run_eval:
            self.evaluate(model)
        return model

    @torch.no_grad()
    def load_model(self):
        modelpool = self.modelpool

        # load models and classification heads
        pretrained_model: ImageEncoder = self.modelpool.load_pretrained_model()
        log.info("pretrained model statistics:")
        print_parameters(pretrained_model, print_fn=log.info)

        finetuned_models: Dict[str, ImageEncoder] = {}
        for model_name in self.modelpool.model_names:
            finetuned_models[model_name] = modelpool.load_model(model_name)

        classification_heads: Dict[str, ClassificationHead] = {}
        for model_name in self.modelpool.model_names:
            classification_heads[model_name] = modelpool.load_classification_head(
                model_name
            )
        self.classification_heads = classification_heads

        self.train_processor = modelpool.train_processor
        self.test_processor = modelpool.test_processor

        with timeit_context("Building the MoE model"):
            model = deepcopy(pretrained_model)

            if self.partial:
                log.info("Weight ensembling only the MLPs")
                # weight ensembling only the MLPs, merge the remaining layers using task arithmetic
                model = task_arithmetic_merge(
                    pretrained_model=model,
                    finetuned_models=list(finetuned_models.values()),
                    scaling_factor=self.init_lambda,
                    inplace=True,
                )

                # fix all parameters
                model.requires_grad_(False)

                for layer_idx in tqdm(
                    range(model.model.visual.transformer.layers), desc="Upscaling MLPs"
                ):
                    resblock: ResidualAttentionBlock = (
                        model.model.visual.transformer.resblocks[layer_idx]
                    )
                    resblock.mlp = ParetoWeightEnsemblingModule(
                        base_model=cast(
                            ResidualAttentionBlock,
                            pretrained_model.model.visual.transformer.resblocks[
                                layer_idx
                            ],
                        ).mlp,
                        expert_models=[
                            cast(
                                ResidualAttentionBlock,
                                m.model.visual.transformer.resblocks[layer_idx],
                            ).mlp
                            for m in finetuned_models.values()
                        ],
                        init_lambda=self.init_lambda,
                        fix_base_model_and_experts=True,
                        router_hidden_layers=self.router_hidden_layers,
                    )
            else:
                log.info("Weight ensembling all the layers")
                # weight ensembling all the layers, merge the remaining layers using task arithmetic
                model = task_arithmetic_merge(
                    pretrained_model=model,
                    finetuned_models=list(finetuned_models.values()),
                    scaling_factor=self.init_lambda,
                    inplace=True,
                )
                # fix all parameters
                model.requires_grad_(False)

                for name in [
                    "conv1",
                    "ln_pre",
                    "ln_post",
                    # "class_embedding",
                    # "positional_embedding",
                ]:
                    setattr(
                        model.model.visual,
                        name,
                        ParetoWeightEnsemblingModule(
                            base_model=getattr(pretrained_model.model.visual, name),
                            expert_models=[
                                getattr(m.model.visual, name)
                                for m in finetuned_models.values()
                            ],
                            init_lambda=self.init_lambda,
                            fix_base_model_and_experts=True,
                            router_hidden_layers=self.router_hidden_layers,
                        ),
                    )
                for layer_idx in tqdm(
                    range(model.model.visual.transformer.layers),
                    desc="Upscaling the transformer layers",
                ):
                    for name in ["ln_1", "attn", "ln_attn", "ln_2", "mlp"]:
                        setattr(
                            model.model.visual.transformer.resblocks[layer_idx],
                            name,
                            ParetoWeightEnsemblingModule(
                                base_model=getattr(
                                    cast(
                                        ResidualAttentionBlock,
                                        pretrained_model.model.visual.transformer.resblocks[
                                            layer_idx
                                        ],
                                    ),
                                    name,
                                ),
                                expert_models=[
                                    getattr(
                                        cast(
                                            ResidualAttentionBlock,
                                            m.model.visual.transformer.resblocks[
                                                layer_idx
                                            ],
                                        ),
                                        name,
                                    )
                                    for m in finetuned_models.values()
                                ],
                                init_lambda=self.init_lambda,
                                fix_base_model_and_experts=True,
                                router_hidden_layers=self.router_hidden_layers,
                            ),
                        )
                for name in ["token_embedding", "ln_final"]:
                    setattr(
                        model.model,
                        name,
                        ParetoWeightEnsemblingModule(
                            base_model=getattr(pretrained_model.model, name),
                            expert_models=[
                                getattr(m.model, name)
                                for m in finetuned_models.values()
                            ],
                            init_lambda=self.init_lambda,
                            fix_base_model_and_experts=True,
                            router_hidden_layers=self.router_hidden_layers,
                        ),
                    )

            self.model = model
            print_parameters(model, print_fn=log.info)
            return model

    def load_datasets(self):
        modelpool = self.modelpool

        # setup the train datasets and loaders
        train_datasets = {}
        train_loaders = {}
        train_loader_iters = {}
        for dataset_name in modelpool.train_dataset_names:
            train_datasets[dataset_name] = modelpool.load_train_dataset(dataset_name)
            train_datasets[dataset_name] = CLIPDataset(
                train_datasets[dataset_name], self.train_processor
            )
            # sanity check
            assert isinstance(train_datasets[dataset_name][0], tuple)

            # setup the train loaders
            train_loaders[dataset_name] = torch.utils.data.DataLoader(
                train_datasets[dataset_name],
                shuffle=True,
                drop_last=True,
                **self._dataloader_kwargs,
            )
            train_loaders[dataset_name] = self.fabric.setup_dataloaders(
                train_loaders[dataset_name]
            )
            train_loaders[dataset_name] = InfiniteDataLoader(
                train_loaders[dataset_name]
            )

            # setup the train loader iterators
            train_loader_iters[dataset_name] = iter(train_loaders[dataset_name])

        self.train_datasets = train_datasets
        self.train_loaders = train_loaders
        self.train_loader_iters = train_loader_iters

        # setup the test datasets and loaders
        test_datasets = {}
        test_loaders = {}
        for dataset_name in modelpool.test_dataset_names:
            test_datasets[dataset_name] = modelpool.load_test_dataset(dataset_name)
            test_datasets[dataset_name] = CLIPDataset(
                test_datasets[dataset_name], self.test_processor
            )
            test_loaders[dataset_name] = torch.utils.data.DataLoader(
                test_datasets[dataset_name],
                shuffle=False,
                **self._dataloader_kwargs,
            )
            test_loaders[dataset_name] = self.fabric.setup_dataloaders(
                test_loaders[dataset_name]
            )

        self.test_datasets = test_datasets
        self.test_loaders = test_loaders

    def compute_loss(self, model: ImageEncoder, ray: Tensor):
        losses = []
        for dataset_idx, dataset_name in enumerate(self.train_datasets):
            batch = next(self.train_loader_iters[dataset_name])
            x, y = batch

            features = model(x)
            logits = self.classification_heads[dataset_name](features)

            _loss = F.cross_entropy(logits, y)
            losses.append(_loss)

        loss = self.aggregate_loss(model, ray, losses)
        return loss

    @abstractmethod
    def aggregate_loss(self, model: nn.Module, ray: Tensor, losses: Tuple[Tensor]):
        pass

    def train(self):
        # setup the model
        num_objectives = len(self.modelpool.model_names)
        model = deepcopy(self.model)
        self.classification_heads = {
            t: h.to(self.fabric.device) for t, h in self.classification_heads.items()
        }

        # set up the optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr,
        )
        model, optimizer = self.fabric.setup(model, optimizer)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.num_steps, eta_min=self.lr * 0.1
        )

        model.train()
        device = self.fabric.device
        for step_idx in tqdm(
            range(1, 1 + self.num_steps), "training", dynamic_ncols=True
        ):
            # sample a preference ray
            ray = torch.from_numpy(
                np.random.dirichlet((self.alpha,) * num_objectives, 1)
                .astype(np.float32)
                .flatten()
            ).to(device)
            ParetoWeightEnsemblingModule.set_preferenece_vector(model, ray)

            loss = self.compute_loss(model, ray)

            optimizer.zero_grad()
            self.fabric.backward(loss)
            optimizer.step()

            lr_scheduler.step()

            self.fabric.log("loss", loss.item(), step=step_idx)

            if step_idx % self.save_interval == 0 or step_idx == self.num_steps:
                ckpt_dir = Path(self.log_dir) / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True, parents=True)
                self.fabric.save(
                    ckpt_dir / f"model_step={step_idx}.ckpt",
                    {"model": model},
                )
        return model

    def evaluate(self, model):
        results = defaultdict(list)

        num_objectives = len(self.modelpool.model_names)
        device = self.fabric.device
        self.classification_heads = {
            t: h.to(self.fabric.device) for t, h in self.classification_heads.items()
        }

        if not lightning.fabric.is_wrapped(model):
            model = self.fabric.setup_module(model)
        model.eval()

        if self.num_evaluation_samples == "equal_weight":
            uniform_grid = np.array(
                [[1 / num_objectives] * num_objectives], dtype=np.float32
            )
        else:
            uniform_grid = generate_simplex_grid(
                num_objectives, self.num_evaluation_samples
            )
        for ray_idx, ray in tqdm(enumerate(uniform_grid), "evaluating samples"):
            results["ray_idx"].append(ray_idx)
            # sample a preference ray
            for i in range(len(ray)):
                results[f"ray_{i}"].append(ray[i])
            ray = torch.from_numpy(ray).to(device)
            ParetoWeightEnsemblingModule.set_preferenece_vector(model, ray)

            accs = []
            for dataset_idx, dataset_name in enumerate(
                tqdm(
                    self.modelpool.test_dataset_names,
                    "evaluating datasets",
                    leave=False,
                )
            ):
                test_loader = self.test_loaders[dataset_name]
                TOTAL_CORRECT = 0
                TOTAL_COUNT = 0
                for batch_idx, batch in enumerate(
                    pbar := tqdm(
                        test_loader,
                        f"evaluate {dataset_name}",
                        leave=False,
                    )
                ):
                    x, y = batch

                    features = model(x)
                    logits = self.classification_heads[dataset_name](features)
                    preds = logits.argmax(-1)

                    correct = (preds == y).sum().item()
                    TOTAL_CORRECT += correct
                    TOTAL_COUNT += len(y)
                    acc = TOTAL_CORRECT / TOTAL_COUNT
                    pbar.set_postfix_str(f"acc={acc:.2f}")

                    if self.quick_evaluation and batch_idx > 20:
                        break
                results[dataset_name].append(acc)
                accs.append(acc)

            # compute the average accuracy
            if "average" not in self.modelpool.test_dataset_names:
                results["average"].append(np.mean(accs))

            (df := pd.DataFrame(results)).to_csv(
                Path(self.log_dir) / "result.csv", index=False
            )
            log.info(df)


class PWEMoELinearScalarizationForOpenCLIP(PWEMoEAlgorithmForOpenCLIP):
    def aggregate_loss(self, model: nn.Module, ray: Tensor, losses: Tuple[Tensor]):
        loss = 0
        for r, l in zip(ray, losses):
            loss += r * l
        return loss


class PWEMoEExactParetoOptimalForOpenCLIP(PWEMoEAlgorithmForOpenCLIP):
    epo_solver: Optional[EPOSolver] = None

    def aggregate_loss(self, model: nn.Module, ray: Tensor, losses: Tuple[Tensor]):
        if self.epo_solver is None:
            num_objectives = len(self.modelpool.model_names)
            self.epo_solver = EPOSolver(n_tasks=num_objectives, n_params=None)
        epo_solver = self.epo_solver

        losses = torch.stack(losses)
        loss = epo_solver.get_weighted_loss(
            losses,
            ray,
            tuple(filter(lambda p: p.requires_grad, model.parameters())),
        )
        return loss
