# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from typing import (
    TYPE_CHECKING,
    Dict,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
)

import numpy as np
from datasets import DatasetDict, Split, load_dataset, load_from_disk
from transformers.utils.versions import require_version

import fusion_bench

from ..extras.constants import FILEEXT2TYPE
from ..extras.misc import has_tokenized_data
from .aligner import align_dataset
from .data_utils import merge_dataset, split_dataset
from .parser import get_dataset_list
from .preprocess import get_preprocess_and_print_func

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import (
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
    )

    from .data_utils import DatasetModule
    from .template import Template


logger = logging.getLogger(__name__)


def load_hf_dataset(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[
        Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]
    ] = None,
    split: Optional[Union[str, Split]] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Loads a single dataset and aligns it to the standard format.
    """
    path, name, data_dir, data_files = None, None, None, None

    dataset = load_dataset(
        path=path,
        name=name,
        data_dir=data_dir,
        data_files=data_files,
        split=split,
        trust_remote_code=True,
        cache_dir=cache_dir,
        **kwargs,
    )

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[
            :target_num
        ]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info(
            "Sampled {} examples from dataset {}.".format(
                dataset_attr.num_samples, dataset_attr
            )
        )

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _get_merged_dataset(
    dataset_names: Optional[Sequence[str]],
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Gets the merged datasets in the standard format.
    """
    if dataset_names is None:
        return None

    datasets = []
    for dataset_attr in get_dataset_list(dataset_names, data_args.dataset_dir):
        if (stage == "rm" and dataset_attr.ranking is False) or (
            stage != "rm" and dataset_attr.ranking is True
        ):
            raise ValueError(
                "The dataset is not applicable in the current training stage."
            )

        datasets.append(
            load_hf_dataset(dataset_attr, model_args, data_args, training_args)
        )

    return merge_dataset(datasets, data_args, seed=training_args.seed)


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Preprocesses the dataset, including format checking and tokenization.
    """
    if dataset is None:
        return None

    preprocess_func, print_function = get_preprocess_and_print_func(
        data_args,
        stage,
        template,
        tokenizer,
        processor,
        do_generate=(training_args.predict_with_generate and is_eval),
    )
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache)
            or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    dataset = dataset.map(
        preprocess_func,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )

    if training_args.should_log:
        try:
            print("eval example:" if is_eval else "training example:")
            print_function(next(iter(dataset)))
        except StopIteration:
            if stage == "pt":
                raise RuntimeError(
                    "Cannot find sufficient samples, consider increasing dataset size."
                )
            else:
                raise RuntimeError(
                    "Cannot find valid samples, check `data/README.md` for the data format."
                )

    return dataset


def get_dataset(
    template: "Template",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    streaming: bool = False,
    tokenized_data_path: Optional[str] = None,
):
    r"""
    Gets the train dataset and optionally gets the evaluation dataset.

    Args:
        tokenizerd_data_path (Optional[str]): Path to save or load the tokenized datasets.
        streaming (bool): Enable dataset streaming.
    """
    # if `tokenized_data_path` is provided, try to load tokenized dataset from disk
    if tokenized_data_path is not None:
        if fusion_bench.utils.path.path_is_dir_and_not_empty(tokenized_data_path):
            # load datasets from disk and return
            logger.warning(
                "Loading dataset from disk will ignore other data arguments."
            )
            dataset_dict: "DatasetDict" = load_from_disk(tokenized_data_path)
            logger.info("Loaded tokenized dataset from {}.".format(tokenized_data_path))

            if streaming:
                dataset_dict = {
                    k: cast(Dataset, v).to_iterable_dataset()
                    for k, v in dataset_dict.items()
                }

            return dataset_dict

        elif streaming:
            # datasets does not exists, will save tokenized datasets later
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset"):
        dataset = _get_merged_dataset(
            data_args.dataset, model_args, data_args, training_args, stage
        )
        eval_dataset = _get_merged_dataset(
            data_args.eval_dataset, model_args, data_args, training_args, stage
        )

    with training_args.main_process_first(desc="pre-process dataset"):
        dataset = _get_preprocessed_dataset(
            dataset,
            data_args,
            training_args,
            stage,
            template,
            tokenizer,
            processor,
            is_eval=False,
        )
        eval_dataset = _get_preprocessed_dataset(
            eval_dataset,
            data_args,
            training_args,
            stage,
            template,
            tokenizer,
            processor,
            is_eval=True,
        )

        if data_args.val_size > 1e-6:
            dataset_dict = split_dataset(dataset, data_args, seed=training_args.seed)
        else:
            dataset_dict = {}
            if dataset is not None:
                if data_args.streaming:
                    dataset = dataset.shuffle(
                        buffer_size=data_args.buffer_size, seed=training_args.seed
                    )

                dataset_dict["train"] = dataset

            if eval_dataset is not None:
                if data_args.streaming:
                    eval_dataset = eval_dataset.shuffle(
                        buffer_size=data_args.buffer_size, seed=training_args.seed
                    )

                dataset_dict["validation"] = eval_dataset

            dataset_dict = DatasetDict(dataset_dict)

        if tokenized_data_path is not None:
            # save tokenized datasets to disk
            dataset_dict.save_to_disk(tokenized_data_path)
            logger.info("Tokenized dataset saved at {}.".format(tokenized_data_path))

        return dataset_dict
