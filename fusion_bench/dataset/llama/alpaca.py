import logging
import os
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset, load_from_disk
from transformers import PreTrainedTokenizer

import fusion_bench

log = logging.getLogger(__name__)


def tokenize_alpaca_dataset(
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


def load_tokenized_alpaca_dataset_from_json(
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
    dataset = tokenize_alpaca_dataset(dataset, tokenizer, max_length=max_length)
    return dataset
