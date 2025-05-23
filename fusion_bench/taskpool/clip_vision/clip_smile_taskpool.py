from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
from transformers.models.clip.modeling_clip import CLIPVisionTransformer

from fusion_bench.method.smile_upscaling import SmileMoELinear
from fusion_bench.models.hf_clip import HFCLIPClassifier

from .taskpool import CLIPVisionModelTaskPool
from .utils.routing_analysis_utils import LayerWiseRoutingWeightSaver


class SmileCLIPVisionModelTaskPool(CLIPVisionModelTaskPool):

    # hooks and handles for saving layer-wise routing weights
    _layer_wise_routing_weights_save_hooks: Dict[Any, LayerWiseRoutingWeightSaver] = {}
    _layer_wise_routing_weights_save_hook_handles: Dict[Any, RemovableHandle] = {}

    def __init__(
        self,
        linear_module_names: Union[List[str], str],
        layer_wise_routing_weights_save_path: Optional[str],
        layer_wise_routing_weights_max_num: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the SMILECLIPVisionModelTaskPool.

        Args:
            linear_module_names (Union[List[str], str]): The names of the linear modules to save the layer-wise routing weights for.
            layer_wise_routing_weights_save_path (Optional[str]): The path to save the layer-wise routing weights.
            layer_wise_routing_weights_max_num (Optional[int]): The maximum number of layer-wise routing weights to save.
        """
        # linear module names
        assert linear_module_names is not None, "linear_module_names must be provided"
        self.linear_module_names = (
            [linear_module_names]
            if isinstance(linear_module_names, str)
            else list(linear_module_names)
        )
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
                for linear_module_name in self.linear_module_names:
                    linear_module = layer.get_submodule(linear_module_name)
                    assert isinstance(
                        linear_module,
                        (SmileMoELinear),
                    ), f"Linear module is expected to be a SmileMoELinear, but got {type(linear_module)}"
                    # layer-wise routing weights
                    hook = LayerWiseRoutingWeightSaver(
                        self.layer_wise_routing_weights_save_path
                        / task_name
                        / f"layer_{i}_{linear_module_name}.pt",
                        max_num=self.layer_wise_routing_weights_max_num,
                    )
                    self._layer_wise_routing_weights_save_hooks[
                        (i, linear_module_name)
                    ] = hook
                    self._layer_wise_routing_weights_save_hook_handles[
                        (i, linear_module_name)
                    ] = linear_module.gate.register_forward_hook(hook)

    def on_task_evaluation_end(self):
        super().on_task_evaluation_end()
        if self.layer_wise_routing_weights_save_path is not None:
            # remove hooks for saving layer-wise routing weights
            for (
                key,
                handle,
            ) in self._layer_wise_routing_weights_save_hook_handles.items():
                self._layer_wise_routing_weights_save_hooks[key].save_routing_weights()
                self._layer_wise_routing_weights_save_hook_handles.pop(key)
                handle.remove()
