import itertools
from typing import Any, Dict, List, Mapping

import numpy as np
from typing_extensions import TYPE_CHECKING

from .arc import Task
from .augmenters import (
    Augmenter,
    Chain,
    Concat,
    Flip,
    IdentityAugmenter,
    IncreaseHeight,
    IncreaseResolution,
    IncreaseWidth,
    PermuteColors,
    PermuteExamples,
    RandomTranslateXY,
    Reflect,
    Repeat,
    Rotate,
    Transpose,
)
from .messagers import MessageRepresenter

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def get_augmenters(
    include_basic: bool = True,
    include_size: bool = True,
    include_chain: bool = True,
    include_repeat: bool = True,
    include_concat: bool = False,
) -> List[Augmenter]:
    basic_augmenters_to_apply = (
        [
            Rotate(90),
            Rotate(270),
            Rotate(180),
            Flip(0),
            Flip(1),
            Reflect(0, reverse=True),
            Reflect(1, reverse=True),
            Reflect(0, reverse=False),
            Reflect(1, reverse=False),
            RandomTranslateXY(),
            Transpose(),
        ]
        if include_basic
        else []
    )

    size_augmenters_to_apply = (
        [
            IncreaseResolution(2),
            IncreaseHeight(2),
            IncreaseWidth(2),
        ]
        if include_size
        else []
    )

    concat_augmenters_to_apply = (
        [
            Concat((IdentityAugmenter(), Rotate(180)), axis=0),
            Concat((IdentityAugmenter(), Rotate(180)), axis=1),
        ]
        if include_concat
        else []
    )

    chain_augmenters_to_apply = (
        [
            Chain([Rotate(90), IncreaseResolution(2)]),
            Chain([Rotate(270), IncreaseResolution(2)]),
            Chain([Rotate(180), IncreaseResolution(2)]),
            Chain([Flip(0), IncreaseResolution(2)]),
            Chain([Flip(1), IncreaseResolution(2)]),
            Chain([Transpose(), IncreaseResolution(2)]),
        ]
        if include_chain
        else []
    )

    repeat_augmenters_to_apply = (
        [
            Repeat(0, 2),
            Repeat(1, 2),
            Repeat(2, 2),
        ]
        if include_repeat
        else []
    )

    augmenters_to_apply = (
        basic_augmenters_to_apply
        + size_augmenters_to_apply
        + concat_augmenters_to_apply
        + chain_augmenters_to_apply
        + repeat_augmenters_to_apply
    )

    print(
        "Augmenters to apply: ", augmenters_to_apply, "len: ", len(augmenters_to_apply)
    )
    return augmenters_to_apply


def format_and_filter(
    formatter: MessageRepresenter,
    tokenizer: "PreTrainedTokenizer",
    task: Task,
):
    """
    Formats and filters a task for model input.

    Args:
        formatter (MessageRepresenter): The formatter to encode the task.
        tokenizer (PreTrainedTokenizer): The tokenizer to tokenize the conversation.
        task: The task to be formatted and filtered.

    Returns:
        Dict[str, Any]: A dictionary containing the formatted data with keys:
            - "input_ids": The tokenized input IDs.
            - "attention_mask": The attention mask for the input IDs.
            - "labels": The labels for the input IDs.
            - "task_id": The task ID.
            - "num_prompt_tokens": The number of prompt tokens.
            - "num_output_tokens": The number of output tokens.
    """
    task_id = task.name
    task = formatter.encode(task)
    conversation = task[0] + [task[1]]
    assert conversation[-1]["role"] == "assistant", "Last message should be assistant"
    prompt_tokens = tokenizer.apply_chat_template(
        conversation[:-1], tokenize=True, add_generation_prompt=True
    )
    generation_tokens = tokenizer.apply_chat_template(conversation, tokenize=True)
    output_tokens = generation_tokens[len(prompt_tokens) :]
    data = {
        "input_ids": prompt_tokens + output_tokens,
        "attention_mask": [1] * len(prompt_tokens) + [1] * len(output_tokens),
        "labels": prompt_tokens + output_tokens,
        "task_id": task_id,
        "num_prompt_tokens": len(prompt_tokens),
        "num_output_tokens": len(output_tokens),
    }
    return data


def get_test_time_train_data(
    original_task: Task,
    augmenters: List[Augmenter],
    n: int = 1,
    permute_n: int = 1,
    seed: int = 0,
) -> List[Task]:
    """
    Generates augmented training data for test-time training.

    Args:
        original_task (Task): The original task containing training examples.
        augmenters (List[Augmenter]): A list of augmenters to apply to the tasks.
        n (int, optional): The number of examples to leave out for testing. Defaults to 1.
        permute_n (int, optional): The number of times to permute the augmented tasks. Defaults to 1.
        seed (int, optional): The random seed for reproducibility. Defaults to 0.

    Returns:
        List[Task]: A list of augmented tasks.
    """
    rng = np.random.RandomState(seed)
    train_examples = original_task.train_examples.copy()
    initial_tasks = []
    N = len(train_examples)
    for i in range(len(train_examples)):
        examples = train_examples.copy()
        indices = set(range(N)) - {i}
        # we already remove i, so we need to remove n-1 more
        combs = list(itertools.combinations(indices, n - 1))
        combs = [indices - set(comb) for comb in combs]

        for comb in combs:
            initial_tasks.append(
                Task(
                    name=original_task.name,
                    train_examples=[examples[j] for j in comb],
                    test_example=examples[i],
                )
            )

    augmented_tasks = []
    for augmenter in augmenters:
        for task in initial_tasks:
            task = augmenter.apply_to_task(task, to_input=True, to_output=True, rng=rng)
            # some augmentations increase shapes
            if not (task.max_height() <= 30 and task.max_width() <= 30):
                continue
            augmented_tasks.append(task)

    augmented_tasks = list(set(augmented_tasks + initial_tasks))

    color_and_permute_augmented_tasks = []

    for _ in range(permute_n):
        for task in augmented_tasks:
            if len(augmenters) != 0:
                new_task = PermuteColors().apply_to_task(
                    task, to_input=True, to_output=True, rng=rng
                )
            else:
                new_task = task
            new_task = PermuteExamples().apply_to_task(
                new_task, rng=rng, to_input=True, to_output=True
            )
            color_and_permute_augmented_tasks.append(new_task)

    augmented_tasks = color_and_permute_augmented_tasks + augmented_tasks
    augmented_tasks = list(set(augmented_tasks))

    return augmented_tasks


def get_formatted_data(
    task: Task,
    augmenters: List[Augmenter],
    formatter: MessageRepresenter,
    tokenizer: "PreTrainedTokenizer",
    leave_n: int = 1,
    permute_n: int = 1,
    seed: int = 0,
    max_tokens: int = 8192,
):
    train_data = get_test_time_train_data(
        task, augmenters, n=leave_n, permute_n=permute_n, seed=seed
    )

    formatted_data = []
    for task in train_data:
        formatted = format_and_filter(formatter, tokenizer, task)
        if len(formatted["input_ids"]) < max_tokens:
            formatted_data.append(formatted)

    return formatted_data


def process_task_for_ttt(
    task: Task,
    augmenters: List[Augmenter],
    formatter: MessageRepresenter,
    tokenizer: "PreTrainedTokenizer",
    permute_n: int = 1,
    Nmax: int = 250,
    seed: int = 0,
):
    rng = np.random.RandomState(seed)

    leave_1_train_data = get_formatted_data(
        task,
        augmenters,
        formatter,
        tokenizer,
        leave_n=1,
        permute_n=permute_n,
        seed=seed,
    )
    leave_2_train_data = get_formatted_data(
        task,
        augmenters,
        formatter,
        tokenizer,
        leave_n=2,
        permute_n=permute_n,
        seed=seed,
    )

    train = leave_1_train_data

    if len(train) == 0:
        train = leave_2_train_data
    elif len(train) < Nmax:
        train += leave_2_train_data[: Nmax - len(train)]
    elif len(train) > Nmax:
        rng.shuffle(train)
        train = train[:Nmax]

    return train


def process_task(
    task: Task,
    formatter: MessageRepresenter,
    tokenizer: "PreTrainedTokenizer",
):
    formatted = format_and_filter(formatter, tokenizer, task)
    return [formatted]
