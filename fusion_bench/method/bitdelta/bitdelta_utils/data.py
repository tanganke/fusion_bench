import torch
from datasets import load_dataset
from transformers import default_data_collator


def _preprocess(tokenizer, examples, max_length=128):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=max_length
    )


def get_dataset(dataset_name: str, subset: str, split: str, size: int = None):
    if size is None:
        dataset = load_dataset(dataset_name, subset)[split]
    else:
        dataset = load_dataset(dataset_name, subset, streaming=True)[split]
        dataset = dataset.take(size)

    return dataset


def get_dataloader(dataset, tokenizer, batch_size, num_workers=4, max_length=128):
    dataset = dataset.map(
        lambda examples: _preprocess(tokenizer, examples, max_length),
        batched=True,
        batch_size=batch_size,
        remove_columns=["text", "timestamp", "url"],
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=default_data_collator,
    )
    return dataloader
