from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def padded_collate_sft(
    batch: List[Dict[str, List[int]]],
    pad_token_id: int = 0,
    input_ids_key: str = "input_ids",
    attention_mask_key: Optional[str] = "attention_mask",
    labels_key: Optional[str] = "labels",
    ignore_idx: int = -100,
) -> Dict[str, torch.Tensor]:
    """
    Pad (right) a batch of sequences to the longest sequence length in the batch, and
    convert integer lists to tensors.

    Args:
        batch (List[Dict[str, List[int]]]): A list of dictionaries containing input, label pairs.
        padding_idx (int): Padding index for input ids. Defaults to 0.
        ignore_idx (int): Padding index for labels. Defaults to -100.

    Returns:
        Dict[str, torch.Tensor]: Collated input and label tensors.
    """
    input_ids = pad_sequence(
        [torch.tensor(x[input_ids_key]) for x in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    if attention_mask_key is not None and attention_mask_key in batch[0]:
        attention_mask = pad_sequence(
            [torch.tensor(x[attention_mask_key]) for x in batch],
            batch_first=True,
            padding_value=0,
        )
    else:
        attention_mask = None

    for i, item in enumerate(batch):
        # if labels_key not in item, copy input_ids to labels_key
        if labels_key not in item:
            item[labels_key] = item[input_ids_key]

    labels = pad_sequence(
        [torch.tensor(x[labels_key]) for x in batch],
        batch_first=True,
        padding_value=ignore_idx,
    )

    if attention_mask is not None:
        collated_batch = {
            input_ids_key: input_ids,
            attention_mask_key: attention_mask,
            labels_key: labels,
        }
    else:
        collated_batch = {input_ids_key: input_ids, labels_key: labels}

    for key in batch[0]:
        if key not in [input_ids_key, attention_mask_key, labels_key]:
            collated_batch[key] = [x[key] for x in batch]

    return collated_batch


def bradley_terry_rm_collate(
    batch: List[Dict[str, List[int]]],
    pad_token_id: int = 0,
    padding_side="right",
):
    """
    Collate function for Bradley-Terry reward modeling.

    Args:
        batch (List[Dict[str, List[int]]]): A list of dictionaries containing input, label pairs.
        pad_token_id (int): Padding index for input ids. Defaults to 0.

    Returns:
        Dict[str, torch.Tensor]: Collated input and label tensors. The first half of the batch is the winner, and the second half is the loser.
    """
    converted_batch = []
    for item in batch:
        new_item = {
            "input_ids": item["chosen_input_ids"],
            "attention_mask": item["chosen_attention_mask"],
        }
        converted_batch.append(new_item)
    for item in batch:
        new_item = {
            "input_ids": item["rejected_input_ids"],
            "attention_mask": item["rejected_attention_mask"],
        }
        converted_batch.append(new_item)

    input_ids = pad_sequence(
        [torch.tensor(x["input_ids"]) for x in converted_batch],
        batch_first=True,
        padding_value=pad_token_id,
        padding_side=padding_side,
    )
    attention_mask = pad_sequence(
        [torch.tensor(x["attention_mask"]) for x in converted_batch],
        batch_first=True,
        padding_value=0,
        padding_side=padding_side,
    )

    collated_batch = {"input_ids": input_ids, "attention_mask": attention_mask}
    for key in batch[0]:
        if key not in [
            "chosen_input_ids",
            "chosen_attention_mask",
            "rejected_input_ids",
            "rejected_attention_mask",
        ]:
            collated_batch[key] = [x[key] for x in batch]
    return collated_batch
