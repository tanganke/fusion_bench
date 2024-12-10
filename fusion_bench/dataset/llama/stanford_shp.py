import os
from copy import deepcopy
from typing import TYPE_CHECKING, Optional

from datasets import Dataset, load_dataset, load_from_disk
from lightning.fabric.utilities import rank_zero_only
from tqdm.auto import tqdm

from fusion_bench.utils import timeit_context

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def load_tokenized_stanford_shp_for_rlhf(
    tokenizer: "PreTrainedTokenizer",
    path: str = "stanfordnlp/SHP",
    split: str = "train",
    num_proc: int = 8,
    cache_path: Optional[str] = None,
):
    if cache_path is not None and os.path.isdir(cache_path):
        dataset = load_from_disk(cache_path)
        return dataset

    dataset = load_dataset(path, split=split)

    def tokenize(sample):
        """
        - history: the post title concatented to the post body (string)
        - human_ref_A: text of comment A (string)
        - human_ref_B: text of comment B (string)
        - labels: the preference label -- it is 1 if A is preferred to B; 0 if B is preferred to A. This was randomized such that the label distribution is roughly 50/50. (integer)
        """
        # Create a conversation with the post title and body, followed by comments
        conversation = [{"role": "user", "content": sample["history"]}]
        if sample["labels"] == 0:
            sample["chosen"] = deepcopy(conversation).append(
                {"role": "assistant", "content": sample["human_ref_B"]}
            )
            sample["rejected"] = deepcopy(conversation).append(
                {"role": "assistant", "content": sample["human_ref_A"]}
            )
        else:
            sample["chosen"] = deepcopy(conversation).append(
                {"role": "assistant", "content": sample["human_ref_A"]}
            )
            sample["rejected"] = deepcopy(conversation).append(
                {"role": "assistant", "content": sample["human_ref_B"]}
            )

        # apply chat template
        sample["chosen_chat"] = tokenizer.apply_chat_template(
            sample["chosen"], tokenize=False, add_generation_prompt=False
        )
        sample["rejected_chat"] = tokenizer.apply_chat_template(
            sample["rejected"], tokenize=False, add_generation_prompt=False
        )

        # tokenize the conversation
        tokenized_pos = tokenizer(sample["chosen_chat"], truncation=True)
        tokenized_neg = tokenizer(sample["rejected_chat"], truncation=True)

        # Ensure that the chosen response does not contain an EOS token
        sample["chosen_input_ids"] = tokenized_pos["input_ids"]
        sample["chosen_attention_mask"] = tokenized_pos["attention_mask"]
        assert (
            tokenizer.eos_token_id not in tokenized_pos["input_ids"][:-1]
        ), f"Prompt contains EOS token: {sample['positive']}"
        if sample["chosen_input_ids"][-1] != tokenizer.eos_token_id:
            sample["chosen_input_ids"].append(tokenizer.eos_token_id)
            sample["chosen_attention_mask"].append(1)

        sample["rejected_input_ids"] = tokenized_neg["input_ids"]
        sample["rejected_attention_mask"] = tokenized_neg["attention_mask"]
        # Ensure that the rejected response does not contain an EOS token
        assert (
            tokenizer.eos_token_id not in tokenized_neg["input_ids"][:-1]
        ), f"Prompt contains EOS token: {sample['rejected']}"
        if sample["rejected_input_ids"][-1] != tokenizer.eos_token_id:
            sample["rejected_input_ids"].append(tokenizer.eos_token_id)
            sample["rejected_attention_mask"].append(1)

        return sample

    dataset = dataset.map(tokenize, num_proc=num_proc)

    if cache_path is not None and rank_zero_only.rank == 0:
        dataset.save_to_disk(cache_path)
    return dataset
