from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def padded_collate_sft(
    batch: List[Dict[str, List[int]]],
    padding_idx: int = 0,
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
        padding_value=padding_idx,
    )
    if attention_mask_key is not None and attention_mask_key in batch[0]:
        attention_mask = pad_sequence(
            [torch.tensor(x[attention_mask_key]) for x in batch],
            batch_first=True,
            padding_value=0,
        )
    else:
        attention_mask = None
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
