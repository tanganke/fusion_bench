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
from enum import Enum, unique
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    TypedDict,
    Union,
)

from datasets import DatasetDict, concatenate_datasets, interleave_datasets

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset


logger = logging.getLogger(__name__)


SLOTS = Sequence[Union[str, Set[str], Dict[str, str]]]


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


def merge_dataset(
    all_datasets: List[Union["Dataset", "IterableDataset"]],
    seed: int,
    mix_strategy: Literal["concat", "interleave_under", "interleave_over"] = "concat",
    streaming: bool = False,
    interleave_probs: Optional[str] = None,
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Merges multiple datasets to a unified dataset.

    Args:
        streaming (bool): Enable dataset streaming.
        interleave_probs (Optional[str]): Probabilities to sample data from datasets. Use commas to separate multiple datasets.
    """
    if len(all_datasets) == 1:
        return all_datasets[0]
    elif mix_strategy == "concat":
        if streaming:
            logger.warning(
                "The samples between different datasets will not be mixed in streaming mode."
            )

        return concatenate_datasets(all_datasets)
    elif mix_strategy.startswith("interleave"):
        if not streaming:
            logger.warning(
                "We recommend using `mix_strategy=concat` in non-streaming mode."
            )

        return interleave_datasets(
            datasets=all_datasets,
            probabilities=interleave_probs,
            seed=seed,
            stopping_strategy=(
                "first_exhausted" if mix_strategy.endswith("under") else "all_exhausted"
            ),
        )
    else:
        raise ValueError("Unknown mixing strategy: {}.".format(mix_strategy))


def split_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    seed: int,
    streaming: bool = False,
    buffer_size: int = 16384,
    val_size: float = 0.0,
) -> "DatasetDict":
    r"""
    Splits the dataset and returns a dataset dict containing train set and validation set.

    Supports both map dataset and iterable dataset.

    Args:
        streaming (bool): Enable dataset streaming.
        buffer_size (bool): Size of the buffer to randomly sample examples from in dataset streaming.
        val_size (flaot): Size of the development set, should be an integer or a float in range `[0,1)`.
    """
    if streaming:
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)
        val_set = dataset.take(int(val_size))
        train_set = dataset.skip(int(val_size))
        return DatasetDict({"train": train_set, "validation": val_set})
    else:
        val_size = int(val_size) if val_size > 1 else val_size
        dataset = dataset.train_test_split(test_size=val_size, seed=seed)
        return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})
