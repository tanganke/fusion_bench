import logging
from typing import Dict, List, cast  # noqa: F401

import torch
import torch.utils.data
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn.modules import Module
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from fusion_bench.dataset.clip_dataset import CLIPDataset
from fusion_bench.mixins import CLIPClassificationMixin

from .regmean import RegMeanAlgorithm

log = logging.getLogger(__name__)


class RegMeanAlgorithmForCLIP(
    RegMeanAlgorithm,
    CLIPClassificationMixin,
):
    _config_mapping = {
        "_dataloader_kwargs": "dataloader_kwargs",
    }

    def __init__(self, *, dataloader_kwargs: DictConfig, **kwargs):
        super().__init__(**kwargs)
        self._dataloader_kwargs = dataloader_kwargs

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
        model: Module,
        train_dataset: torch.utils.data.Dataset,
        linear_modules_to_merge: Dict[str, Module],
    ):
        # setup dataloader
        train_dataset = CLIPDataset(train_dataset, self.clip_processor)
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, **self._dataloader_kwargs
        )
        train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        model = self.fabric.setup(model)

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
        for step, batch in tqdm(
            enumerate(train_dataloader),
            desc=f"computing regmean weights for model {model_name}",
        ):
            if (
                len(num_actual_examples) > 0
                and list(num_actual_examples.values())[0] >= self.num_regmean_examples
            ):
                break
            logits = self.compute_logits(model, batch, model_name)  # noqa: F841

        # remove the added hook
        for handle in handles:
            handle.remove()

        for module_name in regmean_weights.keys():
            regmean_weights[module_name] = regmean_weights[module_name].detach().cpu()

        return regmean_weights
