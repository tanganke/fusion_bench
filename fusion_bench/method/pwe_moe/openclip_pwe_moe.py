from copy import deepcopy
import logging
from typing import Dict, cast, List, Tuple

from omegaconf import DictConfig

from fusion_bench import BaseAlgorithm
from fusion_bench.mixins import OpenCLIPClassificationMixin, SimpleProfilerMixin
from fusion_bench.modelpool import OpenCLIPVisionModelPool
from fusion_bench.models.open_clip import ClassificationHead, ImageEncoder
from fusion_bench.utils import print_parameters, timeit_context

log = logging.getLogger(__name__)


class PWEMoEAlgorithmForOpenCLIP(
    BaseAlgorithm,
    SimpleProfilerMixin,
    OpenCLIPClassificationMixin,
):
    def __init__(
        self,
        *,
        partial: bool,
        init_lambda: float,
        router_hidden_layers: int,
        lr: float,
        num_steps: int,
        save_interval: int,
        alpha: float,
        checkpoint_path: str,
        eval_grid: bool,
        eval_grid_n: int,
        eval_grid_m: int,
        dataloader_kwargs: DictConfig,
        run_train: bool,
        run_eval: bool,
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
        self.eval_grid = eval_grid
        self.eval_grid_n = eval_grid_n
        self.eval_grid_m = eval_grid_m
        self._dataloader_kwargs = dataloader_kwargs
        self.run_train = run_train
        self.run_eval = run_eval

    def run(self, modelpool: OpenCLIPVisionModelPool):
        self.modelpool = modelpool

        self.load_model()
        self.load_datasets()

        if self.run_train:
            self.train()
        if self.run_eval:
            self.evaluate()

    def load_model(self):
        modelpool = self.modelpool

        # load models and classification heads
        pretrained_model = self.modelpool.load_pretrained_model()
        log.info("pretrained model statistics:")
        print_parameters(pretrained_model, print_fn=log.info)

        finetuned_models: Dict[ImageEncoder] = {}
        for model_name in self.modelpool.model_names:
            finetuned_models[model_name] = modelpool.load_model(model_name)

        classification_heads: Dict[ClassificationHead] = {}
        for model_name in self.modelpool.model_names:
            classification_heads[model_name] = modelpool.load_classification_head(
                model_name
            )
        self.classification_heads = classification_heads

        self.train_processor = modelpool.train_processor
        self.test_processor = modelpool.test_processor

        with timeit_context("Building moe model"):
            model = deepcopy(pretrained_model)

            if self.partial:
                # weight ensembling only the MLPs, merge the remaining layers using task arithmetic

                # model fusion
                sd = {}
                base_sd = model.state_dict()
                for name in base_sd.keys():
                    sd[name] = base_sd[name]
                for m in finetuned_models.values():
                    m = cast(ImageEncoder, m)
                    expert_sd = m.state_dict()
                    for name in expert_sd.keys():
                        sd[name] = (
                            sd[name]
                            + (expert_sd[name] - base_sd[name]) * self.init_lambda
                        )
                model.load_state_dict(sd)

                # fix all parameters
                model.requires_grad_(False)

                for layer_idx in range(model.model.visual.transformer.layers):
                    model.model.visual.transformer.resblocks[layer_idx].mlp = (
                        ParetoWeightEnsemblingModule(
                            base_model=cast(
                                ResidualAttentionBlock,
                                self.pretrained_model.model.visual.transformer.resblocks[
                                    layer_idx
                                ],
                            ).mlp,
                            expert_models=[
                                cast(
                                    ResidualAttentionBlock,
                                    m.model.visual.transformer.resblocks[layer_idx],
                                ).mlp
                                for m in self.finetuned_models.values()
                            ],
                            init_lambda=self.cfg.init_lambda,
                            fix_base_model_and_experts=True,
                            router_hidden_layers=self.cfg.router_hidden_layers,
                        )
                    )
            else:
                # weight ensembling all the layers

                # model fusion
                sd = {}
                base_sd = model.state_dict()
                for name in base_sd.keys():
                    sd[name] = base_sd[name]
                for m in self.finetuned_models.values():
                    m = cast(ImageEncoder, m)
                    expert_sd = m.state_dict()
                    for name in expert_sd.keys():
                        sd[name] = (
                            sd[name]
                            + (expert_sd[name] - base_sd[name]) * self.cfg.init_lambda
                        )
                model.load_state_dict(sd)
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
                            base_model=getattr(
                                self.pretrained_model.model.visual, name
                            ),
                            expert_models=[
                                getattr(m.model.visual, name)
                                for m in self.finetuned_models.values()
                            ],
                            init_lambda=self.cfg.init_lambda,
                            fix_base_model_and_experts=True,
                            router_hidden_layers=self.cfg.router_hidden_layers,
                        ),
                    )
                for layer_idx in range(model.model.visual.transformer.layers):
                    for name in ["ln_1", "attn", "ln_attn", "ln_2", "mlp"]:
                        setattr(
                            model.model.visual.transformer.resblocks[layer_idx],
                            name,
                            ParetoWeightEnsemblingModule(
                                base_model=getattr(
                                    cast(
                                        ResidualAttentionBlock,
                                        self.pretrained_model.model.visual.transformer.resblocks[
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
                                    for m in self.finetuned_models.values()
                                ],
                                init_lambda=self.cfg.init_lambda,
                                fix_base_model_and_experts=True,
                                router_hidden_layers=self.cfg.router_hidden_layers,
                            ),
                        )

                for name in ["token_embedding", "ln_final"]:
                    setattr(
                        model.model,
                        name,
                        ParetoWeightEnsemblingModule(
                            base_model=getattr(self.pretrained_model.model, name),
                            expert_models=[
                                getattr(m.model, name)
                                for m in self.finetuned_models.values()
                            ],
                            init_lambda=self.cfg.init_lambda,
                            fix_base_model_and_experts=True,
                            router_hidden_layers=self.cfg.router_hidden_layers,
                        ),
                    )

            self.model = model
            print_parameters(model, print_fn=log.info)

    def load_datasets(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass
