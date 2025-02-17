import logging
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Optional

from datasets import Dataset, load_dataset, load_from_disk
from lightning.fabric.utilities import rank_zero_only
from tqdm.auto import tqdm

from fusion_bench.utils import timeit_context

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

log = logging.getLogger(__name__)


def load_tokenized_preference_700k_for_rlhf(
    tokenizer: "PreTrainedTokenizer",
    path: str = "hendrydong/preference_700K",
    split: str = "train",
    num_proc: int = 8,
    cache_path: Optional[str] = None,
):
    R"""
    Load and tokenized Preference 700k dataset for Bradley-Terry ranking model.

    The returned dataset contains the following fields:

    - chosen_input_ids: The input token ids for the winner.
    - chosen_attention_mask: The attention mask for the winner.
    - rejected_input_ids: The input token ids for the loser.
    - rejected_attention_mask: The attention mask for the loser.
    """
    if cache_path is not None and os.path.exists(cache_path):
        dataset = load_from_disk(cache_path)
        return dataset

    dataset = load_dataset(path, split=split)

    def tokenize(sample):
        sample["chosen_chat"] = tokenizer.apply_chat_template(
            sample["chosen"], tokenize=False, add_generation_prompt=False
        )
        sample["rejected_chat"] = tokenizer.apply_chat_template(
            sample["rejected"], tokenize=False, add_generation_prompt=False
        )

        tokenized_pos = tokenizer(sample["chosen_chat"], truncation=True)
        tokenized_neg = tokenizer(sample["rejected_chat"], truncation=True)

        # Ensure that the chosen response does not contain an PAD token
        sample["chosen_input_ids"] = tokenized_pos["input_ids"]
        sample["chosen_attention_mask"] = tokenized_pos["attention_mask"]
        if tokenizer.pad_token_id in tokenized_pos["input_ids"]:
            log.warning(f"Prompt contains PAD token: {sample['chosen_chat']}")

        sample["rejected_input_ids"] = tokenized_neg["input_ids"]
        sample["rejected_attention_mask"] = tokenized_neg["attention_mask"]
        # Ensure that the rejected response does not contain an PAD token
        if tokenizer.pad_token_id in tokenized_neg["input_ids"]:
            log.warning(f"Prompt contains PAD token: {sample['rejected_chat']}")

        return sample

    dataset = dataset.map(tokenize, num_proc=num_proc)

    if cache_path is not None and rank_zero_only.rank == 0:
        dataset.save_to_disk(cache_path)
    return dataset
