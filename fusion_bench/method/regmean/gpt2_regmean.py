import logging
import os
from copy import deepcopy
from functools import cache
from typing import Dict, List, cast

import lightning as L
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn.modules import Module
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import GPT2ForSequenceClassification, GPT2Model
from transformers.data import default_data_collator
from transformers.models.gpt2.modeling_gpt2 import Conv1D

from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.utils import timeit_context

from .regmean import RegMeanAlgorithm

log = logging.getLogger(__name__)


class RegMeanAlgorithmForGPT2(
    RegMeanAlgorithm,
    LightningFabricMixin,
):
    _include_module_type = [Conv1D]
    classifiers = {}
    _config_mapping = RegMeanAlgorithm._config_mapping | {
        "cache_dir": "cache_dir",
        "batch_size": "batch_size",
        "num_workers": "num_workers",
    }

    def __init__(self, cache_dir: str, batch_size: int, num_workers: int, **kwargs):
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        super().__init__(**kwargs)

    def on_regmean_start(self):
        for model_name in self.modelpool.model_names:
            classifier = cast(
                GPT2ForSequenceClassification,
                self.modelpool.load_classifier(model_name),
            ).requires_grad_(False)
            classifier.transformer = None
            classifier = classifier.to(self.fabric.device)
            self.classifiers[model_name] = classifier

    def compute_logits(self, module: GPT2Model, batch, task: str) -> Tensor:
        self.classifiers[task].transformer = module
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = self.classifiers[task](input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        assert logits.dim() == 2
        return logits

    def get_regmean_weights(
        self,
        model_name: str,
        model: Module,
        train_dataset,
        linear_modules_to_merge: Dict[str, Module],
    ):
        # setup dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=default_data_collator,
            pin_memory=True,
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
                and list(num_actual_examples.values())[0]
                >= self.config.num_regmean_examples
            ):
                break
            logits = self.compute_logits(model, batch, model_name)

        # remove the added hook
        for handle in handles:
            handle.remove()

        for module_name in regmean_weights.keys():
            regmean_weights[module_name] = regmean_weights[module_name].detach().cpu()

        return regmean_weights
