import os
from typing import TYPE_CHECKING, Optional

from datasets import Dataset, load_dataset, load_from_disk
from lightning.fabric.utilities import rank_zero_only
from tqdm.auto import tqdm

from fusion_bench.utils import timeit_context

from .alpaca import convert_alpaca_to_conversation

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def load_tokenized_metamathqa(
    tokenizer: "PreTrainedTokenizer",
    path: str = "meta-math/MetaMathQA",
    split: str = "train",
    cache_path: Optional[str] = None,
):
    if cache_path is not None and os.path.exists(cache_path):
        dataset = load_from_disk(cache_path)
        if split is not None and split in dataset:
            return dataset[split]
        else:
            return dataset

    dataset = load_dataset(path, split=split)

    # convert dataset to alpaca format and save to ../data/MetaMathQA.json
    alpaca_dataset = []
    for example in tqdm(dataset, disable=not rank_zero_only.rank == 0):
        alpaca_example = {
            "instruction": example["query"],
            "input": "",
            "output": example["response"],
        }
        alpaca_dataset.append(alpaca_example)

    conversations = convert_alpaca_to_conversation(alpaca_dataset)
    with timeit_context("Tokenizing dataset"):
        tokenized_dataset = tokenizer.apply_chat_template(
            conversations, return_dict=True
        )
    tokenized_dataset = Dataset.from_dict(tokenized_dataset)

    if cache_path is not None and rank_zero_only.rank == 0:
        tokenized_dataset.save_to_disk(cache_path)
    return tokenized_dataset
