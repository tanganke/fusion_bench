"""
The dataset contains the following fields:

- chosen_input_ids: The input token ids for the winner.
- chosen_attention_mask: The attention mask for the winner.
- rejected_input_ids: The input token ids for the loser.
- rejected_attention_mask: The attention mask for the loser.
"""

import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Subset
from tqdm.auto import tqdm

from fusion_bench.dataset.llama.collate import bradley_terry_rm_collate
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.taskpool import BaseTaskPool
from fusion_bench.utils import instantiate

if TYPE_CHECKING:
    from transformers import LlamaForSequenceClassification


def evaluate_batch(model: "LlamaForSequenceClassification", batch):
    batch_size = batch["input_ids"].size(0)
    assert batch_size % 2 == 0, "Batch size must be even."

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    rewards = outputs[0]
    chosen_reward = rewards[: batch_size // 2]
    rejected_rewards = rewards[batch_size // 2 :]

    loss = -torch.log(torch.sigmoid(chosen_reward - rejected_rewards)).mean()
    correct = (chosen_reward > rejected_rewards).sum().item()
    total = batch_size // 2

    return {
        "loss": loss.item(),
        "correct": correct,
        "total": total,
    }


def evaluate_dataloader(model: "LlamaForSequenceClassification", dataloader):
    """
    Compute the accuracy of the reward model on the given dataloader.

    Args:
        model: The reward model
        dataloader: The dataloader for the dataset

    Returns:
        float: The accuracy of the reward model on the dataset
    """
    metrics = {
        "loss": 0.0,
        "correct": 0,
        "total": 0,
    }
    with torch.no_grad():
        for batch in (pbar := tqdm(dataloader)):
            batch_result = evaluate_batch(model, batch)
            new_total = metrics["total"] + batch_result["total"]
            metrics["loss"] = (
                metrics["loss"] * metrics["total"] / new_total
                + batch_result["loss"] * batch_result["total"] / new_total
            )
            metrics["correct"] += batch_result["correct"]
            metrics["total"] += batch_result["total"]
            pbar.set_postfix(metrics)

    metrics["accuracy"] = metrics["correct"] / metrics["total"]
    return metrics


class RewardModelEvaluationTaskPool(
    BaseTaskPool,
    LightningFabricMixin,
):
    def __init__(
        self,
        test_datasets: List[DictConfig],
        dataloader_kwargs: DictConfig,
        tokenizer: Optional[DictConfig],
        max_num_samples: int = -1,
        seed: int = 0,
        **kwargs,
    ):
        self.seed = seed
        L.seed_everything(seed)
        self._test_datasets = test_datasets
        self.dataloader_kwargs = dataloader_kwargs
        self._tokenizer = tokenizer
        self.max_num_samples = max_num_samples
        super().__init__(**kwargs)

    def setup(self):
        if self._tokenizer is None:
            # try to load the tokenizer from the model pool
            tokenizer = self._program.modelpool.load_tokenizer()
        else:
            tokenizer = instantiate(self._tokenizer)
        self.tokenizer = tokenizer

        test_datasets = {
            dataset_name: instantiate(self._test_datasets[dataset_name])
            for dataset_name in self._test_datasets
        }
        if self.max_num_samples > 0:
            test_datasets = {
                dataset_name: Subset(
                    test_dataset,
                    np.random.permutation(len(test_dataset))[: self.max_num_samples],
                )
                for dataset_name, test_dataset in test_datasets.items()
            }
        test_dataloaders = {
            dataset_name: torch.utils.data.DataLoader(
                test_dataset,
                collate_fn=functools.partial(
                    bradley_terry_rm_collate,
                    pad_token_id=tokenizer.pad_token_id,
                ),
                **self.dataloader_kwargs,
            )
            for dataset_name, test_dataset in test_datasets.items()
        }

        self.test_dataloaders = {
            dataset_name: self.fabric.setup_dataloaders(test_dataloader)
            for dataset_name, test_dataloader in test_dataloaders.items()
        }

    @torch.no_grad()
    def evaluate(self, model: "LlamaForSequenceClassification"):
        self.setup()

        model = self.fabric.setup_module(model)
        if model.config.pad_token_id is None:
            model.config.pad_token_id = self.tokenizer.pad_token_id

        model.eval()
        report = {}
        for dataset_name, test_dataloader in self.test_dataloaders.items():
            report[dataset_name] = evaluate_dataloader(model, test_dataloader)

        print(report)
        return report
