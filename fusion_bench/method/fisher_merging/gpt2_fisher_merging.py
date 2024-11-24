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
from fusion_bench.modelpool import GPT2ForSequenceClassificationPool
from fusion_bench.utils import timeit_context

from .fisher_merging import FisherMergingAlgorithm, get_param_squared_gradients


class FisherMergingAlgorithmForGPT2(
    FisherMergingAlgorithm,
    LightningFabricMixin,
):
    """
    Implements the Fisher Merging Algorithm for GPT-2 models on text classification tasks.

    This class extends the FisherMergingAlgorithm to handle GPT-2 models specifically.
    It supports caching, batch processing, and multi-worker data loading.

    Attributes:
        classifiers (dict): A dictionary to store classifiers for each model.
        modelpool (HuggingFaceGPT2ClassificationPool): The model pool containing the GPT-2 models.
        cache_dir (str): Directory to cache data.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
    """

    classifiers = {}
    modelpool: GPT2ForSequenceClassificationPool = None
    _config_mapping = FisherMergingAlgorithm._config_mapping | {
        "cache_dir": "cache_dir",
        "batch_size": "batch_size",
        "num_workers": "num_workers",
    }

    def __init__(
        self,
        cache_dir: str,
        batch_size: int,
        num_workers: int,
        **kwargs,
    ):
        """
        Initialize the FisherMergingAlgorithmForGPT2 with the given configuration.

        Args:
            cache_dir (str): Directory to cache data.
            batch_size (int): Batch size for data loading.
            num_workers (int): Number of workers for data loading.
            **kwargs: Additional keyword arguments.
        """
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        super().__init__(**kwargs)

    def on_fisher_merging_start(self):
        """
        Setup the classifiers for each model in the model pool before starting the Fisher merging process.
        """
        for model_name in self.modelpool.model_names:
            classifier = cast(
                GPT2ForSequenceClassification,
                self.modelpool.load_classifier(model_name),
            ).requires_grad_(False)
            classifier.transformer = None
            classifier = classifier.to(self.fabric.device)
            self.classifiers[model_name] = classifier

    def compute_logits(self, module: GPT2Model, batch, task: str) -> Tensor:
        """
        Compute the logits for the given batch and task.

        Args:
            module (GPT2Model): The GPT-2 model module.
            batch (dict): The input batch.
            task (str): The name of the task.

        Returns:
            Tensor: The computed logits.
        """
        self.classifiers[task].transformer = module
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = self.classifiers[task](input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        assert logits.dim() == 2
        return logits

    def get_fisher_weights(
        self,
        model_name: str,
        model: Module,
        train_dataset,
        param_names_to_merge: List[str],
    ) -> Dict[str, Tensor]:
        """
        Compute the Fisher weights for the given model and training dataset.

        Args:
            model_name (str): The name of the model.
            model (Module): The model module.
            train_dataset: The training dataset.
            param_names_to_merge (List[str]): List of parameter names to merge.

        Returns:
            Dict[str, Tensor]: The computed Fisher weights for each parameter.
        """
        # setup dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=default_data_collator,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        model = self.fabric.setup(model)
        num_fisher_examples = self.config.num_fisher_examples
        if num_fisher_examples % train_dataloader.batch_size != 0:
            print(
                f"warning: the number of examples for computing fisher cannot be fully divided by the batch size for model, "
                "which may lead to a slightly different number of the actually used examples."
            )
        num_computed_examples = 0
        batches_fisher_weights_list = []
        for step, batch in tqdm(
            enumerate(train_dataloader),
            desc=f"computing fisher weights",
            total=num_fisher_examples // train_dataloader.batch_size,
        ):
            if num_computed_examples >= num_fisher_examples:
                break
            logits = self.compute_logits(model, batch, model_name)
            # Tensor, shape (batch_size, num_label_classes)

            # compute fisher weights for classifxication task
            # use detach() to detach from the computation graph
            # Tensor, shape (batch_size, num_label_classes)
            labels_probabilities = torch.softmax(logits, dim=-1).detach()
            labels_log_probabilities = torch.log_softmax(logits, dim=-1)
            # sqrt labels_probabilities, since torch.sqrt(labels_probabilities) would be squared in the following squared gradients
            labels_expectations = (
                torch.sqrt(labels_probabilities) * labels_log_probabilities
            )
            # sum over label classes and batch dimension
            sum_labels_expectations = labels_expectations.sum(dim=-1).sum(dim=0)
            model.zero_grad()
            sum_labels_expectations.backward()
            # dict, fisher weights of a batch
            batch_fisher_weights = get_param_squared_gradients(
                model=model, param_names_to_merge=param_names_to_merge
            )

            # move fisher weights to cpu to save GPU memory
            for key, weights in batch_fisher_weights.items():
                batch_fisher_weights[key] = weights.detach().cpu()

            batches_fisher_weights_list.append(batch_fisher_weights)
            num_computed_examples += batch["input_ids"].size(0)

        model_to_merge_fisher_weights = {}
        for batch_fisher_weights in batches_fisher_weights_list:
            for key in batch_fisher_weights:
                if key not in model_to_merge_fisher_weights:
                    model_to_merge_fisher_weights[key] = batch_fisher_weights[key]
                else:
                    model_to_merge_fisher_weights[key] += batch_fisher_weights[key]

        # mean over batches
        for key in model_to_merge_fisher_weights:
            model_to_merge_fisher_weights[key] /= num_computed_examples
            model_to_merge_fisher_weights[key] = (
                model_to_merge_fisher_weights[key].detach().cpu()
            )
        return model_to_merge_fisher_weights
