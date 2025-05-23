from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
from transformers.models.clip.modeling_clip import CLIPVisionTransformer

from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.models.sparse_we_moe import (
    SparseWeightEnsemblingMoE,
    SparseWeightEnsemblingMoE_ShardGate,
)

from .taskpool import CLIPVisionModelTaskPool
from .utils.routing_analysis_utils import LayerWiseRoutingWeightSaver


class SparseWEMoECLIPVisionModelTaskPool(CLIPVisionModelTaskPool):

    # hooks and handles for saving layer-wise routing weights
    _layer_wise_routing_weights_save_hooks: Dict[Any, LayerWiseRoutingWeightSaver] = {}
    _layer_wise_routing_weights_save_hook_handles: Dict[Any, RemovableHandle] = {}

    _config_mapping = CLIPVisionModelTaskPool._config_mapping | {
        "_layer_wise_routing_weights_save_path": "layer_wise_routing_weights_save_path",
    }

    def __init__(
        self,
        layer_wise_routing_weights_save_path: Optional[str],
        layer_wise_routing_weights_max_num: Optional[int] = None,
        **kwargs,
    ):
        # save path for layer-wise routing weights
        self._layer_wise_routing_weights_save_path = (
            layer_wise_routing_weights_save_path
        )
        self.layer_wise_routing_weights_save_path = (
            Path(layer_wise_routing_weights_save_path)
            if layer_wise_routing_weights_save_path is not None
            else None
        )
        self.layer_wise_routing_weights_max_num = layer_wise_routing_weights_max_num
        super().__init__(**kwargs)

    def on_task_evaluation_begin(self, classifier: HFCLIPClassifier, task_name: str):
        super().on_task_evaluation_begin(classifier, task_name)
        if self.layer_wise_routing_weights_save_path is not None:
            # setup hooks for saving layer-wise routing weights
            assert isinstance(
                classifier.clip_model.vision_model,
                (CLIPVisionTransformer, CLIPVisionModel),
            ), "Vision model is expected to be a CLIPVisionTransformer"
            vision_model = classifier.clip_model.vision_model
            if isinstance(vision_model, CLIPVisionModel):
                vision_model = vision_model.vision_model
                # assign forward hooks for each layer
            shared_gate = None
            for i, layer in enumerate(vision_model.encoder.layers):
                mlp = layer.mlp
                assert isinstance(
                    mlp,
                    (SparseWeightEnsemblingMoE, SparseWeightEnsemblingMoE_ShardGate),
                ), f"MLP is expected to be a SparseWeightEnsemblingMoE or SparseWeightEnsemblingMoE_ShardGate, but got {type(mlp)}"
                # layer-wise routing weights
                hook = LayerWiseRoutingWeightSaver(
                    self.layer_wise_routing_weights_save_path
                    / task_name
                    / f"layer_{i}.pt",
                    max_num=self.layer_wise_routing_weights_max_num,
                )
                self._layer_wise_routing_weights_save_hooks[i] = hook
                if isinstance(mlp, SparseWeightEnsemblingMoE_ShardGate):
                    # if use shared gate, copy the gate to all layers to avoid multiple hooks
                    if shared_gate is None:
                        shared_gate = mlp.gate
                    mlp.gate = deepcopy(shared_gate)
                self._layer_wise_routing_weights_save_hook_handles[i] = (
                    mlp.gate.register_forward_hook(hook)
                )

    def on_task_evaluation_end(self):
        super().on_task_evaluation_end()
        if self.layer_wise_routing_weights_save_path is not None:
            # remove hooks for saving layer-wise routing weights
            for i, handle in self._layer_wise_routing_weights_save_hook_handles.items():
                self._layer_wise_routing_weights_save_hooks[i].save_routing_weights()
                self._layer_wise_routing_weights_save_hook_handles.pop(i)
                handle.remove()
