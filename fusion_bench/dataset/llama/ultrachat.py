import os
from typing import TYPE_CHECKING, Optional

from datasets import Dataset, load_dataset, load_from_disk
from lightning.fabric.utilities import rank_zero_only
from tqdm.auto import tqdm

from fusion_bench.utils import timeit_context

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def load_tokenized_ultrachat_200k(
    tokenizer: "PreTrainedTokenizer",
    path: str = "HuggingFaceH4/ultrachat_200k",
    split: str = "train_sft",
    num_proc: int = 8,
    cache_path: Optional[str] = None,
):
    R"""
    Load and tokenized Ultrachat 200k dataset for Bradley-Terry ranking model.

    The returned dataset contains the following fields:

    - input_ids: The input token ids for the winner.
    - attention_mask: The attention mask for the winner.
    """
    if cache_path is not None and os.path.exists(cache_path):
        dataset = load_from_disk(cache_path)
        return dataset

    dataset = load_dataset(path, split=split)

    def tokenize(sample):

        # ? is it necessary to `.replace(tokenizer.bos_token, "")`?
        sample["input_ids"] = tokenizer.apply_chat_template(
            sample["messages"], tokenize=True, add_generation_prompt=False
        )
        sample["attention_mask"] = [1] * len(sample["input_ids"])

        return sample

    dataset = dataset.map(tokenize, num_proc=num_proc)

    if cache_path is not None and rank_zero_only.rank == 0:
        dataset.save_to_disk(cache_path)
    return dataset


if __name__ == "__main__":
    # Example usage and testing
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    dataset = load_tokenized_ultrachat_200k(tokenizer)
    print(dataset)
