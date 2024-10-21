from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
from transformers.models.clip.modeling_clip import CLIPVisionTransformer

from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.models.sparse_we_moe import SparseWeightEnsemblingMoE

from .taskpool import CLIPVisionModelTaskPool


class LayerWiseRoutingWeightSaver:
    def __init__(self, save_path: Path, max_num: Optional[int] = None):
        self.save_path = save_path
        self.routing_weights = []
        self.max_num = max_num

    def __call__(self, module, input, output: Tuple[Tensor]):
        routing_weights = output[0].detach().cpu()
        if self.max_num is not None and self.max_num > 0:
            if len(self.routing_weights) > self.max_num:
                return
            elif routing_weights.size(0) + len(self.routing_weights) > self.max_num:
                self.routing_weights.append(
                    routing_weights[: self.max_num - len(self.routing_weights)]
                )
            else:
                self.routing_weights.append(routing_weights)
        else:
            self.routing_weights.append(routing_weights)

    def save_routing_weights(self):
        routing_weights = torch.cat(self.routing_weights, dim=0)
        if self.save_path is not None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving routing weights to {self.save_path}")
            torch.save(routing_weights, self.save_path)


class SparseWEMoECLIPVisionModelTaskPool(CLIPVisionModelTaskPool):

    # hooks and handles for saving layer-wise routing weights
    _layer_wise_routing_weights_save_hooks = {}
    _layer_wise_routing_weights_save_hook_handles: Dict[int, RemovableHandle] = {}

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
            for i, layer in enumerate(vision_model.encoder.layers):
                mlp = layer.mlp
                assert isinstance(mlp, SparseWeightEnsemblingMoE)
                hook = LayerWiseRoutingWeightSaver(
                    self.layer_wise_routing_weights_save_path
                    / task_name
                    / f"layer_{i}.pt",
                    max_num=self.layer_wise_routing_weights_max_num,
                )
                self._layer_wise_routing_weights_save_hooks[i] = hook
                self._layer_wise_routing_weights_save_hook_handles[i] = (
                    mlp.gate.register_forward_hook(hook)
                )

    def on_task_evaluation_end(self):
        super().on_task_evaluation_end()
        if self.layer_wise_routing_weights_save_path is not None:
            # remove hooks for saving layer-wise routing weights
            for i, handle in self._layer_wise_routing_weights_save_hook_handles.items():
                handle.remove()
                self._layer_wise_routing_weights_save_hooks[i].save_routing_weights()
