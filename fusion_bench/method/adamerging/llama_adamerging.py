import logging
from pathlib import Path

import torch

from fusion_bench import BaseModelFusionAlgorithm
from fusion_bench.mixins import LightningFabricMixin, SimpleProfilerMixin
from fusion_bench.modelpool import CausalLMPool
from fusion_bench.models.wrappers.layer_wise_fusion import (
    LayerWiseMergedModel,
    get_layer_wise_weights,
)
from fusion_bench.utils.data import load_tensor_from_file

log = logging.getLogger(__name__)


class LayerWiseAdaMergingForLlama(
    BaseModelFusionAlgorithm,
    LightningFabricMixin,
    SimpleProfilerMixin,
):

    def __init__(
        self,
        optimizer: str,
        lr: float,
        init_values: 0.3,
        init_weights_path: str,
        clamp_weights: bool,
        max_steps: int,
        dataloader_kwargs: bool,
        tie_weights: bool,
        strict: bool,
        **kwargs,
    ):
        self.optimizer = optimizer
        self.lr = lr
        self.init_values = init_values
        self.init_weights_path = init_weights_path
        self.clamp_weights = clamp_weights
        self.max_steps = max_steps
        self.tie_weights = tie_weights
        self.strict = strict
        self.dataloader_kwargs = dataloader_kwargs
        super().__init__(**kwargs)

    def run(self, modelpool: CausalLMPool):
        assert (
            modelpool.has_pretrained
        ), "Must be a pre-tarined model with name `_pretrained_` in the model pool."
        log.info(f"There are {len(modelpool)} expert models in the model pool.")

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
            clamp_weights=self.config.clamp_weights,
            tie_weights=self.config.tie_weights,
            strict=self.config.strict,
        )
        print(f"{layer_wise_weight.size()=}, {layer_wise_weight.numel()=}")
        return module
