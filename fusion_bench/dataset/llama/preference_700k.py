import os
from typing import TYPE_CHECKING, Optional

from datasets import Dataset, load_dataset, load_from_disk
from lightning.fabric.utilities import rank_zero_only
from tqdm.auto import tqdm

from fusion_bench.utils import timeit_context

from .alpaca import convert_alpaca_to_conversation

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def load_tokenized_preference_700k_for_bradley_terry_rm(
    tokenizer: "PreTrainedTokenizer",
    path: str = "hendrydong/preference_700K",
    split: str = "train",
    num_proc: int = 8,
    cache_path: Optional[str] = None,
):
    R"""
    Load and tokenized Preference 700k dataset for Bradley-Terry ranking model.

    The returned dataset contains the following fields:

    - input_ids_j: The input token ids for the winner.
    - attention_mask_j: The attention mask for the winner.
    - input_ids_k: The input token ids for the loser.
    - attention_mask_k: The attention mask for the loser.
    """
    if cache_path is not None and os.path.exists(cache_path):
        dataset = load_from_disk(cache_path)
        return dataset

    dataset = load_dataset(path, split=split)

    def tokenize(sample):

        # ? is it necessary to `.replace(tokenizer.bos_token, "")`?
        sample["positive"] = tokenizer.apply_chat_template(
            sample["chosen"], tokenize=False, add_generation_prompt=False
        ).replace(tokenizer.bos_token, "")
        sample["negative"] = tokenizer.apply_chat_template(
            sample["rejected"], tokenize=False, add_generation_prompt=False
        ).replace(tokenizer.bos_token, "")

        tokenized_pos = tokenizer(sample["positive"], truncation=True)
        tokenized_neg = tokenizer(sample["negative"], truncation=True)
        sample["input_ids_j"] = tokenized_pos["input_ids"]
        sample["attention_mask_j"] = tokenized_pos["attention_mask"]
        sample["input_ids_k"] = tokenized_neg["input_ids"]
        sample["attention_mask_k"] = tokenized_neg["attention_mask"]
        return sample

    dataset = dataset.map(tokenize, num_proc=num_proc)

    if cache_path is not None and rank_zero_only.rank == 0:
        dataset.save_to_disk(cache_path)
    return dataset
