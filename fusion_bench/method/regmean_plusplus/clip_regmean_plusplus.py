import logging
from collections import defaultdict
from typing import Dict, List, cast  # noqa: F401

import torch
import torch.utils.data
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn.modules import Module
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from fusion_bench.dataset.clip_dataset import CLIPDataset
from fusion_bench.mixins import CLIPClassificationMixin, auto_register_config
from fusion_bench.mixins.clip_classification import (
    VISION_MODEL_PREFIX,
    vision_backbone,
)

from .regmean_plusplus import RegMeanAlgorithmPlusPlus

log = logging.getLogger(__name__)


@auto_register_config
class RegMeanAlgorithmForCLIPPlusPlus(
    RegMeanAlgorithmPlusPlus,
    CLIPClassificationMixin,
):
    def __init__(self, *, dataloader_kwargs: DictConfig, **kwargs):
        super().__init__(**kwargs)

    def on_regmean_start(self):
        self.setup_zero_shot_classification_head()

    def compute_logits(self, module, batch, task: str) -> Tensor:
        images, _ = batch
        text_embeds = self.zeroshot_weights[task]

        image_embeds = module(images)[1]
        image_embeds = self.visual_projection(image_embeds)

        # normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity
        logits_per_text = (
            torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale_exp
        )
        logits_per_image = logits_per_text.t()

        return logits_per_image

    def get_regmean_weights(
        self,
        model_name: str,
        layer: Module,
        batches_input: List[Tensor],
        linear_modules_to_merge: Dict[str, Module],
    ):
        layer = self.fabric.setup(layer)

        def compute_regmean_weights(module_name: str):
            """
            compute the regmean weights, a hook function to deal with each module's input
            :param module_name: str, module name
            :return:
            """

            def hook(module: nn.Module, input: tuple, output: torch.Tensor):
                # Tensor, shape (batch_size, sequence_length, hidden_dim)
                x = cast(Tensor, input[0]).detach()
                batch_num_actual_examples = x.shape[0]
                # Tensor, shape (batch_size * sequence_length, hidden_dim)
                x = x.reshape(-1, x.shape[-1])
                # Tensor, shape (hidden_dim, hidden_dim)
                xtx = torch.matmul(x.transpose(0, 1), x)
                # store the averaged weights in regmean_weights
                if module_name not in regmean_weights.keys():
                    regmean_weights[module_name] = xtx / x.shape[0]
                    num_computed_examples[module_name] = x.shape[0]
                    num_actual_examples[module_name] = batch_num_actual_examples
                else:
                    regmean_weights[module_name] = (
                        regmean_weights[module_name]
                        * num_computed_examples[module_name]
                        + xtx
                    ) / (num_computed_examples[module_name] + x.shape[0])
                    num_computed_examples[module_name] += x.shape[0]
                    num_actual_examples[module_name] += batch_num_actual_examples

            return hook

        handles = []
        # dictionary, regmean matrices for each linear module inputs
        regmean_weights = {}
        # dictionary, number of examples (multiplied the sequence length) used for computing regmean matrices
        num_computed_examples = {}
        # dictionary, number of actual examples used for computing regmean matrices
        num_actual_examples = {}

        for module_name, linear_module_to_merge in linear_modules_to_merge.items():
            # register a hook in the forward process
            handle = linear_module_to_merge.register_forward_hook(
                compute_regmean_weights(module_name=module_name)
            )
            handles.append(handle)
        _ = self.layer_batches_forward(layer, batches_input)

        # remove the added hook
        for handle in handles:
            handle.remove()

        for module_name in regmean_weights.keys():
            regmean_weights[module_name] = regmean_weights[module_name].detach().cpu()

        return regmean_weights

    def merge_embedding_layer(self, models_to_merge_dict: Dict[str, nn.Module]):
        models_to_merge_param_dict = defaultdict(list)

        # get the parameters of the embedding layer from each model
        for model_to_merge in models_to_merge_dict.values():
            model_to_merge_state_dict = model_to_merge.state_dict()

            param_dict = {}
            for name, param in model_to_merge_state_dict.items():
                if name.startswith(
                    f"{VISION_MODEL_PREFIX}embeddings"
                ) or name.startswith(f"{VISION_MODEL_PREFIX}pre_layrnorm"):
                    param_dict[name] = param

            for param_name in param_dict.keys():
                models_to_merge_param_dict[param_name].append(param_dict[param_name])

        # merge the parameters of the embedding layer
        merged_params_dict = {}
        for param_name, param_list in models_to_merge_param_dict.items():
            merged_params_dict[param_name] = torch.stack(param_list).mean(dim=0)

        return merged_params_dict

    def get_input_for_first_layer(self, model: nn.Module, train_dataset):
        # setup dataloader
        train_dataset = CLIPDataset(train_dataset, self.clip_processor)
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, **self.dataloader_kwargs
        )
        train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        model = self.fabric.setup(model)

        def compute_input(model, batch):
            images, _ = batch

            images = images.to(model.device)
            backbone = vision_backbone(model)
            image_embeds = backbone.embeddings(images)
            image_embeds = backbone.pre_layrnorm(image_embeds)
            image_embeds = image_embeds.detach().cpu()

            return image_embeds

        num_computed_examples = 0
        num_regmean_examples = self.num_regmean_examples

        batches_input = []
        for batch in train_dataloader:
            if num_computed_examples >= num_regmean_examples:
                break
            batches_input.append(compute_input(model, batch))
            num_computed_examples += batch[0].size(0)

        return batches_input

    def get_layers(self, model: nn.Module):
        return vision_backbone(model).encoder.layers

    def update_merged_params_dict(
        self, merged_params_dict, new_merged_params, layer_idx
    ):
        for key, value in new_merged_params.items():
            key = f"{VISION_MODEL_PREFIX}encoder.layers.{layer_idx}.{key}"
            merged_params_dict[key] = value

        return merged_params_dict

    def layer_batches_forward(
        self, layer: nn.Module, batches_input: List[Tensor]
    ) -> Tensor:
        batches_output = []
        for batch in batches_input:
            device = next(layer.parameters()).device
            batch = batch.to(device)
            # `causal_attention_mask` is a required, no-default param on
            # `CLIPEncoderLayer.forward` in `transformers < 5`; passing it here is a
            # harmless no-op on `transformers >= 5`, which absorbs it into
            # `**kwargs: Unpack[TransformersKwargs]` instead of declaring it -- so
            # always passing it (rather than gating it on `VISION_MODEL_PREFIX`)
            # keeps this call valid on both.
            output = layer(batch, attention_mask=None, causal_attention_mask=None)
            # `transformers < 5` returns `(hidden_states, ...)`, so `[0]` unpacked the
            # tuple; `transformers >= 5` returns a bare tensor, where indexing `[0]`
            # would instead slice away the batch dimension -- so check the type rather
            # than gating on the version flag, since this needs no version lookup to
            # get right.
            if isinstance(output, tuple):
                output = output[0]
            logits = output.detach().cpu()
            batches_output.append(logits)
        return batches_output
