R"""
This is an experimental implementation of an adaptive SVD-based merging algorithm for CLIP vision encoders.

This implementation is based on the following paper:

    - Ante Tang et al. SMILE: Zero-Shot Sparse Mixture of Low-Rank Experts Construction From Pre-Trained Foundation Models. Aug, 2024. http://arxiv.org/abs/2408.10174

After the upscaled model is created, the algorithm reduces and unload the MoE modules to the original linear modules by averaging the routing weights of the experts.
"""

from copy import deepcopy
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union, cast

import lightning as L
import torch
import torch.nn.functional as F
import torch.utils.hooks
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, random_split
from tqdm.auto import tqdm
from transformers import CLIPVisionModel
from transformers.models.clip.modeling_clip import (
    CLIPEncoderLayer,
    CLIPVisionTransformer,
)
from typing_extensions import override

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.dataset import CLIPDataset
from fusion_bench.method import WeightedAverageAlgorithm
from fusion_bench.method.simple_average import simple_average
from fusion_bench.mixins import SimpleProfilerMixin
from fusion_bench.modelpool import CLIPVisionModelPool
from fusion_bench.models.smile_moe.linear_from_module import (
    ExpertNotTrainedError,
    SmileMoELinear,
)
from fusion_bench.models.utils import find_layers_with_type, get_attr, set_attr
from fusion_bench.utils.devices import get_device


class AvgRoutingWeightsMetric:
    def __init__(
        self,
        linear: SmileMoELinear,
    ):
        super().__init__()
        self.linear = linear
        self.reset()

    def reset(self):
        linear = self.linear
        self.sample_count = 0
        self.accumulated_routing_weights = torch.zeros(
            linear.num_experts,
            dtype=torch.float32,
            device=get_device(linear),
        )

    def __call__(
        self,
        linear: SmileMoELinear,
        inps: Tuple[Tensor],
        out: Tensor,
    ):
        assert len(inps) == 1
        inp = inps[0]
        inp = inp.view(-1, inp.size(-1))

        # compute routing weights
        router_logits = linear.gate(inp)
        routing_weights = F.softmax(router_logits, dim=-1)

        # accumulate routing weights
        self.accumulated_routing_weights += routing_weights.sum(dim=0)
        self.sample_count += inp.size(0)

    def compute(self):
        return self.accumulated_routing_weights / self.sample_count


class AdaSVDMergingForCLIPVisionModel(
    BaseAlgorithm,
    SimpleProfilerMixin,
):
    _linear_layer_cls = (nn.Linear,)

    def __init__(
        self,
        scaling_factor: Optional[Union[float, List[float]]],
        num_samples: int,  # budget of the number of samples to be used
        gate_k: int,
        average_experts: bool,  # if True, the non-linear modules of experts are averaged. Otherwise, copy the pretrained one.
        device: Optional[Literal["cuda", "cpu"]],
        upscaling_accelerator: str,
        seed: Optional[int],
        **kwargs,
    ):
        if seed is not None:
            L.seed_everything(seed)
        self.scaling_factor = scaling_factor
        self.num_samples = num_samples
        self.gate_k = gate_k
        self.average_experts = average_experts
        if device is not None:
            self.device = device
        else:
            self.device = "cpu"  # default device
        self.upscaling_accelerator = upscaling_accelerator
        super().__init__(**kwargs)

    @override
    @torch.no_grad()
    def run(self, modelpool: CLIPVisionModelPool):
        # check preconditions
        if isinstance(self.scaling_factor, Iterable):
            assert len(modelpool.model_names) == len(
                self.scaling_factor
            ), "The number of scaling factors should be equal to the number of expert models."

        # setup the data and model
        dataset = self.prepare_data(modelpool)
        upscaled_model = self.prepare_model(modelpool)  # SMILE Upscaling

        # forward pass
        vision_model: CLIPVisionTransformer = upscaled_model.vision_model
        all_hidden_states = []
        for sample_idx, sample in enumerate(
            tqdm(dataset, desc="Extracting hidden states to the first layer")
        ):
            image, _ = sample
            hidden_states = vision_model.embeddings(image.unsqueeze(0).to(self.device))
            hidden_states = vision_model.pre_layrnorm(hidden_states)
            all_hidden_states.append(hidden_states)
        all_hidden_states = torch.concat(all_hidden_states, dim=0)

        # reduce the MoE modules to the original linear modules
        for layer_idx, layer in enumerate(
            tqdm(vision_model.encoder.layers, desc="Reduce MoE to Linear")
        ):
            layer = cast(CLIPEncoderLayer, layer)

            linear_layers = cast(
                Dict[str, SmileMoELinear],
                find_layers_with_type(layer, layer_types=[SmileMoELinear]),
            )

            def get_hook_fn(linear: SmileMoELinear):
                hook_fn = AvgRoutingWeightsMetric(linear)
                return hook_fn

            hooks: Dict[str, AvgRoutingWeightsMetric] = {}
            handles: List[torch.utils.hooks.RemovableHandle] = []
            for name, linear in linear_layers.items():
                hook_fn = get_hook_fn(linear)
                hooks[name] = hook_fn
                handles.append(linear.register_forward_hook(hook_fn))

            # forward pass
            hidden_states_to_next_layer = []
            for sample_idx, _ in enumerate(all_hidden_states):
                hidden_states = all_hidden_states[sample_idx : sample_idx + 1]
                hidden_states = layer(
                    hidden_states,
                    attention_mask=None,
                    causal_attention_mask=None,
                    output_attentions=False,
                )[0]
                hidden_states_to_next_layer.append(hidden_states)
            all_hidden_states = torch.concat(hidden_states_to_next_layer, dim=0)

            # compute merge weights and merge
            average_routing_weights = {}
            for name, hook_fn in hooks.items():
                average_routing_weights[name] = hook_fn.compute()
            for h in handles:
                h.remove()

            # merge the MoE modules to the original linear modules
            for name, linear in linear_layers.items():
                weights: Tensor = average_routing_weights[name].cpu()
                if self.scaling_factor is not None:
                    if isinstance(self.scaling_factor, (int, float)):
                        weights = weights * self.scaling_factor
                    elif isinstance(self.scaling_factor, Iterable):
                        weights = weights * torch.asarray(self.scaling_factor)
                    else:
                        raise ValueError(
                            f"scaling_factor should be int, float, or a list/tuple of float. Got {type(self.scaling_factor)}"
                        )
                source_models = {"pretrained_model": linear.pretrained_model} | {
                    f"expert_{i}": expert for i, expert in enumerate(linear.experts)
                }
                merged_linear = WeightedAverageAlgorithm(
                    normalize=False, weights=[1] + weights.tolist(), verbose=False
                ).run(BaseModelPool(source_models))
                set_attr(layer, name.split("."), merged_linear)

        return upscaled_model

    def prepare_data(self, modelpool: CLIPVisionModelPool):
        all_datasets = []
        processor = modelpool.load_processor()
        for dataset_name in modelpool.train_dataset_names:
            dataset = modelpool.load_train_dataset(dataset_name)
            assert len(dataset) >= self.num_samples, (
                f"Number of samples in the dataset ({len(dataset)}) "
                f"should be greater than or equal to the budget ({self.num_samples})"
            )
            dataset = random_split(
                dataset, [self.num_samples, len(dataset) - self.num_samples]
            )[0]
            dataset = CLIPDataset(dataset, processor)
            all_datasets.append(dataset)
        dataset = ConcatDataset(all_datasets)
        return dataset

    def prepare_model(self, modelpool: CLIPVisionModelPool):
        with self.profile("load pretrained model"):
            pretrained_model: CLIPVisionModel = modelpool.load_pretrained_model()
        with self.profile("load fine-tuned model"):
            finetuned_models: List[CLIPVisionModel] = [
                m for m in tqdm(modelpool.models(), total=len(modelpool.model_names))
            ]

        if self.device == "cuda" and torch.cuda.is_available():
            pretrained_model = pretrained_model.cuda()
            finetuned_models = [m.cuda() for m in finetuned_models]

        with self.profile("merge model"):
            # SMILE upscaling, with dense experts
            upscaled_model = self.merge(pretrained_model, finetuned_models)
        return upscaled_model

    def merge(
        self,
        pretrained_model: nn.Module,
        finetuned_models: List[nn.Module],
        in_place: bool = True,
    ):
        """
        Merges the pretrained model with the fine-tuned models to create an upscaled model.

        Args:
            pretrained_model (nn.Module): The pretrained model.
            finetuned_models (List[nn.Module]): A list of fine-tuned models.
            in_place (bool): If True, modifies the pretrained model in place. Otherwise, creates a copy.

        Returns:
            nn.Module: The merged model.
        """
        if in_place:
            model = pretrained_model
        else:
            model = deepcopy(pretrained_model)

        self._upscale_submodules(model, finetuned_models)
        return model

    def _upscale_linear_layer(
        self,
        pretrained_model,
        finetuned_models,
        name: str,
    ):
        name_list = name.split(".")
        module = get_attr(pretrained_model, name_list)
        experts = [get_attr(m, name_list) for m in finetuned_models]
        try:
            moe_linear = SmileMoELinear(
                module,
                experts,
                gate_k=self.gate_k,
                k=-1,  # we set k to -1 to use dense experts
                top_k=len(finetuned_models),
                routing_use_diff=True,
                full_matrices=True,
                upscaling_accelerator=self.upscaling_accelerator,
            )
        except ExpertNotTrainedError:
            print(f"skip {name} because the experts are not trained.")
            return
        set_attr(pretrained_model, name_list, moe_linear)
        # remove the original module from fine-tuned models to save memory
        for m in finetuned_models:
            set_attr(m, name_list, None)

    def _average_experts(self, pretarined_model, finetuned_models, name: str):
        name_list = name.split(".")
        experts = [get_attr(m, name_list) for m in finetuned_models]
        averaged_module = simple_average(experts)
        set_attr(pretarined_model, name_list, averaged_module)

    def _upscale_submodules(
        self,
        pretrained_model: nn.Module,
        finetuned_models: List[nn.Module],
        tqdm_desc: str = "Upscaling Linear Modules",
    ):
        """
        Upscales the submodules of the pretrained model by merging them with the corresponding submodules from the fine-tuned models.

        Args:
            pretrained_model (nn.Module): The pretrained model.
            finetuned_models (List[nn.Module]): A list of fine-tuned models.
            tqdm_desc (str): Description for the tqdm progress bar.
        """
        for name, module in tqdm(
            tuple(pretrained_model.named_modules()),
            tqdm_desc,
            leave=False,
            dynamic_ncols=True,
        ):
            if isinstance(module, self._linear_layer_cls):
                self._upscale_linear_layer(
                    pretrained_model=pretrained_model,
                    finetuned_models=finetuned_models,
                    name=name,
                )
            elif self.average_experts and len(tuple(module.named_modules())) == 1:
                # if the module is a leaf module, we perform a parameter average
                self._average_experts(pretrained_model, finetuned_models, name)
