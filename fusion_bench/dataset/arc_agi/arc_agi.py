import functools
import multiprocessing
import os
import sys
from multiprocessing import Pool
from typing import Any, Dict, List, Literal, Optional

import fusion_bench
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from lightning.fabric.utilities import rank_zero_only
from tqdm.auto import tqdm
from typing_extensions import TYPE_CHECKING

from .arc import Example, Task
from .preprocess import get_augmenters, process_task

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def _get_formatter(format: Literal["new", "barc"]):
    from . import messagers, representers

    if format == "new":
        standard_formatter = representers.TextTaskRepresenter(
            example_representer=representers.TextExampleRepresenter(
                io_sep=" -> ",
                input_header="",
                output_header="",
                output_footer="#",
                grid_representer=representers.PythonListGridRepresenter(),
            )
        )
        formatter = messagers.GPTTextMessageRepresenterV2(
            task_representer=standard_formatter
        )
    elif format == "barc":
        formatter = messagers.GPTTextMessageRepresenterForBarc(
            prompt=(
                "Cutting Knowledge Date: December 2023\n"
                "Today Date: 26 Jul 2024\n\n"
                "You are a world-class puzzle solver with exceptional pattern recognition skills. "
                "Your task is to analyze puzzles, spot patterns, and provide direct solutions."
            ),
            task_representer=representers.TextTaskRepresenter(
                example_representer=representers.TextExampleRepresenter(
                    grid_representer=representers.WordGridRepresenter(),
                    input_header="Input:\n",
                    output_header="\nOutput:\n",
                    io_sep="\n",
                )
            ),
        )
    else:
        formatter = messagers.GPTTextMessageRepresenterV2()

    return formatter


def _join_list(lists: List[List[Any]]) -> List[Any]:
    ans = []
    for l in lists:
        ans.extend(l)
    return ans


def _to_task(
    train_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    name: str,
) -> List[Task]:
    # train examples
    train_examples = [
        Example(input=np.asarray(data["input"]), output=np.asarray(data["output"]))
        for data in train_data
    ]
    # test example
    test_examples = [
        Example(input=np.asarray(data["input"]), output=np.asarray(data["output"]))
        for data in test_data
    ]
    tasks = [
        Task(train_examples=train_examples, test_example=test_example, name=name)
        for test_example in test_examples
    ]
    return tasks


def _tokenizer_tasks(
    tasks: List[Task],
    tokenizer: "PreTrainedTokenizer",
    use_data_augmentation: bool = True,
    permute_n: int = 1,
    seed: int = 0,
):
    if not use_data_augmentation:
        augmenters_to_apply = []
    else:
        augmenters_to_apply = get_augmenters(
            include_basic=True,
            include_size=True,
            include_chain=True,
            include_repeat=True,
        )

    formatter = _get_formatter("new")
    processor = functools.partial(
        process_task,
        augmenters=augmenters_to_apply,
        formatter=formatter,
        tokenizer=tokenizer,
        permute_n=permute_n,
        Nmax=250,
        seed=seed,
    )

    # with Pool(multiprocessing.cpu_count()) as p:
    #     data = p.map(processor, tasks)
    data = _join_list(
        [
            processor(task)
            for task in tqdm(
                tasks,
                desc="Processing tasks",
                dynamic_ncols=True,
                leave=False,
                disable=not rank_zero_only.rank == 0,
            )
        ]
    )
    dataset = Dataset.from_list(data)
    return dataset


def load_tokenized_arc_agi_dataset(
    tokenizer: Optional["PreTrainedTokenizer"],
    path: str = "dataartist/arc-agi",
    split: Optional[str] = None,
    cache_path: Optional[str] = None,
    use_data_augmentation: bool = True,
    permute_n: int = 1,
    seed: int = 0,
    max_num_tasks: Optional[int] = None,
):
    # regularize split
    if split.lower() == "train":
        split = "training"
    if split.lower() == "test":
        split = "evaluation"

    # load cached dataset if available
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

    # load raw dataset
    datasets = load_dataset(path, split=split)
    if split is None:
        converted_datasets = {
            "train": _join_list(
                [
                    _to_task(
                        task["train"],
                        task["test"],
                        task["id"],
                    )
                    for task in datasets["training"]
                ]
            ),
            "test": _join_list(
                [
                    _to_task(
                        task["train"],
                        task["test"],
                        task["id"],
                    )
                    for task in datasets["evaluation"]
                ]
            ),
        }
        if max_num_tasks is not None:
            # limit the number of tasks, useful for debugging
            converted_datasets = {
                split: converted_datasets[split][:max_num_tasks]
                for split in converted_datasets
            }
        converted_datasets = {
            split: _tokenizer_tasks(
                converted_datasets[split],
                tokenizer,
                use_data_augmentation,
                permute_n,
                seed,
            )
            for split in tqdm(
                converted_datasets,
                desc="Processing splits",
                dynamic_ncols=True,
                disable=not rank_zero_only.rank == 0,
            )
        }
        converted_datasets = DatasetDict(converted_datasets)
    else:
        converted_datasets = _join_list(
            [
                _to_task(
                    task["train"],
                    task["test"],
                    task["id"],
                )
                for task in datasets
            ]
        )
        if max_num_tasks is not None:
            # limit the number of tasks, useful for debugging
            converted_datasets = converted_datasets[:max_num_tasks]
        converted_datasets = _tokenizer_tasks(
            converted_datasets, tokenizer, use_data_augmentation, permute_n, seed
        )

    if cache_path is not None:
        os.makedirs(cache_path, exist_ok=True)
        converted_datasets.save_to_disk(cache_path)
    return converted_datasets
