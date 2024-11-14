import logging
import os
from typing import Any, Dict, List, Literal, Optional

from datasets import load_dataset, load_from_disk
from transformers import PreTrainedTokenizer

import fusion_bench

log = logging.getLogger(__name__)


def load_tokenized_squad_dataset(
    tokenizer: Optional[PreTrainedTokenizer],
    path: Literal["squard_v2", "squad"] = "squard_v2",
    split: Optional[str] = None,
    max_length: int = 384,  # The maximum length of a feature (question and context)
    doc_stride: int = 128,  # The authorized overlap between two part of the context when splitting it is needed.
    datasets: Optional[Any] = None,
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

    # 1. load raw dataset
    if datasets is not None:
        log.info("Use `datasets`, `path` is ignored.")
    else:
        datasets = load_dataset(path)

    # 2. tokenize the dataset
    pad_on_right = tokenizer.padding_side == "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Initialize arrays for start and end positions
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            # Get corresponding example from the original dataset
            sample_idx = sample_mapping[i]
            answer = examples["answers"][sample_idx]

            # Character start/end positions of the answer
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])

            # Convert character positions to token positions
            # Find start token position
            token_start_index = 0
            while (
                token_start_index < len(offset)
                and offset[token_start_index][0] <= start_char
            ):
                token_start_index += 1
            token_start_index -= 1

            # Find end token position
            token_end_index = token_start_index
            while (
                token_end_index < len(offset) and offset[token_end_index][1] <= end_char
            ):
                token_end_index += 1
            token_end_index -= 1

            start_positions.append(token_start_index)
            end_positions.append(token_end_index)

        tokenized_examples["start_positions"] = start_positions
        tokenized_examples["end_positions"] = end_positions

        return tokenized_examples

    tokenized_datasets = datasets.map(
        prepare_train_features,
        batched=True,
        remove_columns=datasets["train"].column_names,
    )

    if cache_path is not None:
        os.makedirs(cache_path, exist_ok=True)
        tokenized_datasets.save_to_disk(cache_path)

    if split is None:
        return tokenized_datasets
    else:
        return tokenized_datasets[split]
