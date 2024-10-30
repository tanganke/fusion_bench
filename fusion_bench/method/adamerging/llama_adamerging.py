import logging
from pathlib import Path

import torch
import os

from fusion_bench import BaseModelFusionAlgorithm
from fusion_bench.mixins import LightningFabricMixin, SimpleProfilerMixin
from fusion_bench.modelpool import CausalLMPool
from fusion_bench.models.wrappers.layer_wise_fusion import (
    LayerWiseMergedModel,
    get_layer_wise_weights,
)
from fusion_bench.utils import instantiate
from fusion_bench.utils.data import load_tensor_from_file
import lightning as L
from fusion_bench.utils.parameters import print_parameters

log = logging.getLogger(__name__)


class LayerWiseAdaMergingForLlamaSFT(
    BaseModelFusionAlgorithm,
    LightningFabricMixin,
    SimpleProfilerMixin,
):

    modelpool: CausalLMPool

    def __init__(
        self,
        seed: int,
        output_dir: str,
        optimizer: str,
        lr: float,
        init_values: 0.3,
        init_weights_path: str,
        clamp_weights: bool,
        max_steps: int,
        dataloader_kwargs: bool,
        tie_weights: bool,
        strict: bool,
        skip_training: bool = False,
        **kwargs,
    ):
        """
        Args:
            seed (int): random seed to set at the begining of running.
        """
        self.seed = seed
        self.output_dir = output_dir
        self.optimizer = optimizer
        self.lr = lr
        self.init_values = init_values
        self.init_weights_path = init_weights_path
        self.clamp_weights = clamp_weights
        self.max_steps = max_steps
        self.tie_weights = tie_weights
        self.strict = strict
        self.dataloader_kwargs = dataloader_kwargs
        self.skip_training = skip_training
        super().__init__(**kwargs)

    def run(self, modelpool: CausalLMPool):
        self.modelpool = modelpool
        fabric = self.fabric

        assert (
            modelpool.has_pretrained
        ), "Must be a pre-tarined model with name `_pretrained_` in the model pool."
        log.info(f"There are {len(modelpool)} expert models in the model pool.")

        fabric.seed_everything(self.seed)

        if fabric.global_rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)

        with self.profiler("construct_layer_wise_merged_model"):
            module = self.construct_layer_wise_merged_model(modelpool)
            print_parameters(module)

        if not self.skip_training:
            module = self.train(module)

        model = module.merge_and_unload()
        return model

    @torch.no_grad()
    def construct_layer_wise_merged_model(self, modelpool: CausalLMPool):
        """
        Constructs a wrapped layer-wise merged model from model pool.

        This method creates a new wrapped model by merging the layers of a pretrained model with those of several fine-tuned models.
        The merging is controlled by layer-wise weights, which is a `torch.Tensor` of the shape `(num_models, num_layers)`.
        The merging weights can be initialized based on a provided configuration or loaded from a file.

        Args:
            modelpool (ModelPool): An object containing the pretrained model and fine-tuned models to be merged.

        Returns:
            LayerWiseMergedModel: An instance of the merged model with layer-wise weights applied.
        """
        pretrained_model = modelpool.load_model("_pretrained_")
        finetuned_models = [
            modelpool.load_model(name) for name in modelpool.model_names
        ]

        # initialize layer-wise weights using the provided configuration `init_values` or load from file if `weights` is provided
        if self.init_weights_path is None:
            layer_wise_weight = get_layer_wise_weights(
                num_models=len(modelpool.model_names),
                num_layers=len(
                    tuple(
                        filter(lambda p: p.requires_grad, pretrained_model.parameters())
                    )
                ),
                init_values=self.init_values,
            )
        else:
            if isinstance(self.init_weights_path, (str, Path)):
                # self.config.weights is a path to a saved tensor
                layer_wise_weight = load_tensor_from_file(self.init_weights_path)
            else:
                raise ValueError(
                    f"Unsupported weights format: {self.init_weights_path}"
                )

        module = LayerWiseMergedModel(
            layer_wise_weight=layer_wise_weight,
            pretrained_model=pretrained_model,
            finetuned_models=finetuned_models,
            clamp_weights=self.clamp_weights,
            tie_weights=self.tie_weights,
            strict=self.strict,
        )
        print(f"{layer_wise_weight.size()=}, {layer_wise_weight.numel()=}")
        return module

    def train(self, module: LayerWiseMergedModel):
        tokenizer = self.modelpool.load_tokenizer()
        