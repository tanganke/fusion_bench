import functools
import logging
import os

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import GPT2ForSequenceClassification, GPT2Model
from transformers.data import default_data_collator

from fusion_bench.dataset import CLIPDataset
from fusion_bench.modelpool import HuggingFaceGPT2ClassificationPool
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.utils import timeit_context

from .layer_wise_gossip import LayerWiseGossipAlgorithm
from typing import Dict, List, cast

log = logging.getLogger(__name__)

class InfiniteDataLoader:
    """
    A wrapper class for DataLoader to create an infinite data loader.
    This is useful in case we are only interested in the number of steps and not the number of epochs.

    This class wraps a DataLoader and provides an iterator that resets
    when the end of the dataset is reached, creating an infinite loop.

    Attributes:
        data_loader (DataLoader): The DataLoader to wrap.
        data_iter (iterator): An iterator over the DataLoader.
    """

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)  # Reset the data loader
            data = next(self.data_iter)
        return data


class GPT2LayerWiseGossipAlgorithm(LayerWiseGossipAlgorithm):
    """
    A class for layer-wise adaptive merging of gpt models.

    This class extends the LayerWiseGossipAlgorithm to provide specific
    functionaluty for GPT2 models, including loading datasets, construction classification
    layers, and computing logits.
    """

    modelpool: HuggingFaceGPT2ClassificationPool
    scores = {}

    def __init__(self, algorithm_config: DictConfig):
        super().__init__(algorithm_config)

    @functools.cache
    def get_test_dataset(self, task: str):
        """
        load the test dataset for the task.
        This method is cached, so the dataset is loaded only once.

        Args:
            task (str): The name of the task.

        Return:
            Dataset: The test dataset for the task.
        """

        log.info(f"Loading test dataset: {task}")
        dataset = self.modelpool.load_test_dataset(task)
        return dataset

    @functools.cache
    def get_shuffled_test_loader_iter(self, task: str) -> DataLoader:
        """
        Get an iterator over the shuffled test DataLoader for the task.

        Args:
            task (str): The name of the task.

        Returns:
            iterator: An iterator over the shuffled test DataLoader.
        """
        loader = DataLoader(
            self.get_test_dataset(task),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=default_data_collator,
            pin_memory=True,
        )

        if self.fabric is not None:
            loader = self.fabric.setup_dataloaders(loader)
        return iter(InfiniteDataLoader(loader))

    def on_test_time_adaptation_start(self):
        """
        Prepare for test-time adaptation.
        """
        if isinstance(self.scores, dict) and self.scores:
            return
        for model_name in self.modelpool.model_names:
            score = cast(
                GPT2ForSequenceClassification,
                self.modelpool.load_classifier(model_name),
            ).score.requires_grad_(False)
            score = score.to(self.fabric.device)
            self.scores[model_name] = score
        

    def compute_logits(self, module: GPT2Model, batch, task: str) -> Tensor:
        """
        Compute the logits for the given batch and task.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_size, _ = input_ids.shape[:2]
        pad_token_id = 50256

        transformer_outputs = module(
            input_ids,
            past_key_values=None,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
        )
        hidden_states = transformer_outputs[0]
        logits = self.scores[task](hidden_states)

        sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(logits.device)

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        assert pooled_logits.dim() == 2
        return pooled_logits

        