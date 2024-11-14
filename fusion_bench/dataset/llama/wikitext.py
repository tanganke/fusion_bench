import logging
import os
from typing import Any, Dict, List, Optional

from datasets import load_dataset, load_from_disk
from transformers import PreTrainedTokenizer

import fusion_bench

log = logging.getLogger(__name__)


def load_tokenized_wiki_dataset(
    tokenizer: Optional[PreTrainedTokenizer],
    path: str = "wikitext",
    name: str = "wikitext-2-raw-v1",
    split: Optional[str] = None,
    datasets: Optional[Any] = None,
    block_size: int = 128,
    cache_path: Optional[str] = None,
):
    """
    Reference: https://github.com/huggingface/notebooks/blob/main/examples/language_modeling.ipynb

    Args:
        block_size (int):
        dataset: If dataset is provided, `path` and `name` will be ignored.
    """
    if cache_path is not None and fusion_bench.utils.path.path_is_dir_and_not_empty(
        cache_path
    ):
        datasets = load_from_disk(cache_path)
        if split is None:
            return datasets
        else:
            return datasets[split]
    else:
        assert (
            tokenizer is not None
        ), "Cached dataset not found. Need tokenizer to process the raw data."

    # 1. load raw dataset
    if datasets is not None:
        log.info("Use `datasets`, `path` and `name` are ignored.")
    else:
        datasets = load_dataset(path, name)

    # 2. tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = datasets.map(
        tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
    )

    # If we now look at an element of our datasets, we will see the text have been replaced by the input_ids the model will need:
    # { 'attention_mask': <list of int>, 'input_ids': <list of int> }

    # 3. concat and truncate tokens
    def group_texts(examples: Dict[str, List]):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    if cache_path is not None:
        os.makedirs(cache_path, exist_ok=True)
        lm_datasets.save_to_disk(cache_path)

    if split is None:
        return lm_datasets
    else:
        return lm_datasets[split]
