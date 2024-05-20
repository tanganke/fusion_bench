"""
This module provides a class to load and cache GLUE datasets for GPT-2 models.

Examples:

```python
from transformers import GPT2Tokenizer
from fusion_bench.dataset.gpt2_glue import TokenizedGLUE

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
glue = TokenizedGLUE(tokenizer)
dataset = glue.load_dataset("mrpc")
"""

from functools import partial
from pathlib import Path
from typing import Literal

from datasets import load_dataset, load_from_disk


def cache_dataset(
    func,
    model_name="gpt2",
    cache_dir: str | Path = "outputs",
):
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    def wrapper(*args, **kwargs):
        cache_path = cache_dir / model_name / f"_{func.__name__}_cached"
        if cache_path.parent.exists() is False:
            cache_path.parent.mkdir(parents=True)
        if cache_path.exists():
            dataset = load_from_disk(str(cache_path))
        else:
            dataset = func(*args, **kwargs)
            dataset.save_to_disk(str(cache_path.absolute()))
        return dataset

    return wrapper


# Tokenize and convert examples to features
def mrpc_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def mnli_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def cola_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def qnli_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["question"],
        examples["sentence"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def qqp_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["question1"],
        examples["question2"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


class TokenizedGLUE:
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def load_dataset(
        self, name: Literal["mrpc", "mnli", "cola", "sst2", "qnli", "qqp", "rte"]
    ):
        glue_dataset_loaders = {
            "mrpc": self.load_mrpc_dataset,
            "mnli": self.load_mnli_dataset,
            "cola": self.load_cola_dataset,
            "sst2": self.load_sst2_dataset,
            "qnli": self.load_qnli_dataset,
            "qqp": self.load_qqp_dataset,
            "rte": self.load_rte_dataset,
            # "wnli": load_wnli_dataset,
        }
        return glue_dataset_loaders[name]()

    @cache_dataset
    def load_mrpc_dataset(self):
        dataset = load_dataset("glue", "mrpc")
        dataset = dataset.map(
            partial(mrpc_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence1", "sentence2"],
        )
        return dataset

    @cache_dataset
    def load_rte_dataset(self):
        dataset = load_dataset("glue", "rte")
        dataset = dataset.map(
            # RTE has the same format as MRPC
            partial(mrpc_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence1", "sentence2"],
        )
        return dataset

    @cache_dataset
    def load_wnli_dataset(self):
        dataset = load_dataset("glue", "wnli")
        dataset = dataset.map(
            partial(mrpc_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence1", "sentence2"],
        )
        return dataset

    @cache_dataset
    def load_qqp_dataset(self):
        dataset = load_dataset("glue", "qqp")
        dataset = dataset.map(
            partial(qqp_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["question1", "question2"],
        )
        return dataset

    @cache_dataset
    def load_mnli_dataset(self):
        dataset = load_dataset("glue", "mnli")
        dataset = dataset.map(
            partial(mnli_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["premise", "hypothesis"],
        )
        return dataset

    @cache_dataset
    def load_cola_dataset(self):
        dataset = load_dataset("glue", "cola")
        dataset = dataset.map(
            partial(cola_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence"],
        )
        return dataset

    @cache_dataset
    def load_sst2_dataset(self):
        dataset = load_dataset("glue", "sst2")
        dataset = dataset.map(
            partial(cola_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence"],
        )
        return dataset

    @cache_dataset
    def load_qnli_dataset(self):
        dataset = load_dataset("glue", "qnli")
        dataset = dataset.map(
            partial(qnli_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["question", "sentence"],
        )
        return dataset
