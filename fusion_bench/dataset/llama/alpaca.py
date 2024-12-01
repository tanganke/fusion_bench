import logging
import os
import warnings
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset, load_from_disk
from lightning.fabric.utilities import rank_zero_only
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

import fusion_bench
from fusion_bench.utils import timeit_context

log = logging.getLogger(__name__)


def convert_alpaca_to_conversation(alpaca_data: List[Dict[str, str]]):
    """
    Convert Alpaca format data to conversation format.

    Args:
        alpaca_data (list): List of dictionaries in Alpaca format with
            'instruction', 'input', and 'output' keys

    Returns:
        list: List of conversations in ChatML format
    """
    conversations = []

    for item in tqdm(
        alpaca_data,
        "Converting Alpaca to conversations",
        disable=not rank_zero_only.rank == 0,
    ):
        # Skip if required fields are missing
        if not item.get("instruction") or not item.get("output"):
            continue

        conversation = []

        # Create user message
        user_content = item["instruction"]
        if item.get("input") and item["input"].strip():
            user_content += f"\n\n{item['input']}"

        conversation.append({"role": "user", "content": user_content})

        # Create assistant message
        conversation.append({"role": "assistant", "content": item["output"]})

        conversations.append(conversation)

    return conversations


def load_tokenized_alpaca_dataset(
    tokenizer: PreTrainedTokenizer,
    path: str = "yahma/alpaca-cleaned",
    split: str = "train",
    cache_path: Optional[str] = None,
):
    """
    Load and tokenized Alpaca dataset and Alpaca-like dataset.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the dataset.
        path (str, optional): The path to the Alpaca dataset. Defaults to "yahma/alpaca-cleaned".
        split (str, optional): The dataset split to load (e.g., "train", "test"). Defaults to "train".
        cache_path (Optional[str], optional): The path to cache the tokenized dataset. If provided and the cache exists,
            the dataset will be loaded from the cache. Defaults to None.

    Returns:
        Dataset: The tokenized dataset.
    """
    if cache_path is not None and os.path.exists(cache_path):
        dataset = load_from_disk(cache_path)
        if split is not None and split in dataset:
            return dataset[split]
        else:
            return dataset

    dataset = load_dataset(path, split=split)

    alpaca_data = dataset.to_list()
    conversations = convert_alpaca_to_conversation(alpaca_data)
    with timeit_context("Tokenizing dataset"):
        tokenized_dataset = tokenizer.apply_chat_template(
            conversations, return_dict=True
        )
    tokenized_dataset = Dataset.from_dict(tokenized_dataset)

    if cache_path is not None and rank_zero_only.rank == 0:
        tokenized_dataset.save_to_disk(cache_path)
    return tokenized_dataset


def _tokenize_alpaca_dataset_with_template(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    input_template: str = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    input_no_template: str = "### Instruction:\n{instruction}\n\n### Response:\n",
    batch_size: int = 1000,
) -> Dataset:
    """
    Tokenize Alpaca format dataset with customizable options in batches.

    Args:
        dataset: The input dataset in Alpaca format
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        input_template: Template for samples with input field
        input_no_template: Template for samples without input field
        batch_size: Size of batches to process at once

    Returns:
        Tokenized dataset
    """
    warnings.warn(
        "This function is deprecated. Use `apply_chat_template` from `transformers` instead.",
        DeprecationWarning,
    )

    def prepare_samples(samples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        # Format prompts based on whether input field exists
        prompts = []
        for instruction, input_text in zip(
            samples["instruction"], samples.get("input", [])
        ):
            if input_text.strip():
                prompt = input_template.format(
                    instruction=instruction.strip(), input=input_text.strip()
                )
            else:
                prompt = input_no_template.format(instruction=instruction.strip())
            prompts.append(prompt)

        responses = [output.strip() for output in samples["output"]]

        # Tokenize prompts and responses
        prompt_tokens = tokenizer(
            prompts, add_special_tokens=False, padding=False, truncation=False
        )
        response_tokens = tokenizer(
            responses, add_special_tokens=False, padding=False, truncation=False
        )

        input_ids, labels = [], []

        # Process each sample in the batch
        for prompt_toks, response_toks in zip(
            prompt_tokens["input_ids"], response_tokens["input_ids"]
        ):
            # Create input_ids with EOS token
            sample_input_ids = prompt_toks + response_toks + [tokenizer.eos_token_id]

            # Create labels: -100 for prompt, actual tokens for response
            label = [-100] * len(prompt_toks) + response_toks + [tokenizer.eos_token_id]

            # Truncate if exceeds max length
            if len(sample_input_ids) > max_length:
                sample_input_ids = sample_input_ids[:max_length]
                label = label[:max_length]

            input_ids.append(sample_input_ids)
            labels.append(label)

        # Use tokenizer's padding feature for input_ids and attention_mask
        padded_results = tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            max_length=max_length,
            return_attention_mask=True,
        )

        # Pad labels with -100
        padded_labels = []
        for label in labels:
            padding_length = max_length - len(label)
            if padding_length > 0:
                label = label + [-100] * padding_length
            padded_labels.append(label)

        return {
            "input_ids": padded_results["input_ids"],
            "attention_mask": padded_results["attention_mask"],
            "labels": padded_labels,
        }

    if tokenizer.pad_token is None:
        log.warning("Tokenizer does not have a `pad_token`. Set it the `eos_token`.")
        tokenizer.pad_token = tokenizer.eos_token

    # Process the entire dataset in batches
    tokenized_dataset = dataset.map(
        prepare_samples,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    return tokenized_dataset


def load_tokenized_alpaca_dataset_from_json_with_prompt(
    data_files: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    split: Optional[str] = "train",
    cache_path: Optional[str] = None,
):
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

    dataset = load_dataset("json", data_files=data_files)
    if split is not None:
        dataset = dataset[split]
    dataset = _tokenize_alpaca_dataset_with_template(
        dataset, tokenizer, max_length=max_length
    )
    return dataset
