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
from transformers import PreTrainedTokenizer


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
    """
    A class to load and cache GLUE datasets for GPT-2 models.

    This class provides methods to load various GLUE datasets and tokenize them
    using a provided tokenizer. The datasets are cached to disk to avoid
    reloading and tokenizing them multiple times.

    Attributes:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the datasets.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Initialize the TokenizedGLUE class with a tokenizer.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the datasets.
        """
        super().__init__()
        self.tokenizer = tokenizer

    def load_dataset(
        self, name: Literal["mrpc", "mnli", "cola", "sst2", "qnli", "qqp", "rte"]
    ):
        """
        Load and tokenize a GLUE dataset.

        This method loads a specified GLUE dataset, tokenizes it using the provided
        tokenizer, and caches the tokenized dataset to disk.

        Args:
            name (Literal["mrpc", "mnli", "cola", "sst2", "qnli", "qqp", "rte"]): The name of the GLUE dataset to load.

        Returns:
            Dataset: The tokenized GLUE dataset.
        """
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
        """
        Load and tokenize the MRPC dataset.

        This method loads the MRPC dataset, tokenizes it using the provided
        tokenizer, and caches the tokenized dataset to disk.

        Returns:
            Dataset: The tokenized MRPC dataset.
        """
        dataset = load_dataset("glue", "mrpc")
        dataset = dataset.map(
            partial(mrpc_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence1", "sentence2"],
        )
        return dataset

    @cache_dataset
    def load_rte_dataset(self):
        """
        Load and tokenize the RTE dataset.

        This method loads the RTE dataset, tokenizes it using the provided
        tokenizer, and caches the tokenized dataset to disk.

        Returns:
            Dataset: The tokenized RTE dataset.
        """
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
        """
        Load and tokenize the WNLI dataset.

        This method loads the WNLI dataset, tokenizes it using the provided
        tokenizer, and caches the tokenized dataset to disk.

        Returns:
            Dataset: The tokenized WNLI dataset.
        """
        dataset = load_dataset("glue", "wnli")
        dataset = dataset.map(
            partial(mrpc_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence1", "sentence2"],
        )
        return dataset

    @cache_dataset
    def load_qqp_dataset(self):
        """
        Load and tokenize the QQP dataset.

        This method loads the QQP dataset, tokenizes it using the provided
        tokenizer, and caches the tokenized dataset to disk.

        Returns:
            Dataset: The tokenized QQP dataset.
        """
        dataset = load_dataset("glue", "qqp")
        dataset = dataset.map(
            partial(qqp_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["question1", "question2"],
        )
        return dataset

    @cache_dataset
    def load_mnli_dataset(self):
        """
        Load and tokenize the MNLI dataset.

        This method loads the MNLI dataset, tokenizes it using the provided
        tokenizer, and caches the tokenized dataset to disk.

        Returns:
            Dataset: The tokenized MNLI dataset.
        """
        dataset = load_dataset("glue", "mnli")
        dataset = dataset.map(
            partial(mnli_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["premise", "hypothesis"],
        )
        return dataset

    @cache_dataset
    def load_cola_dataset(self):
        """
        Load and tokenize the CoLA dataset.

        This method loads the CoLA dataset, tokenizes it using the provided
        tokenizer, and caches the tokenized dataset to disk.

        Returns:
            Dataset: The tokenized CoLA dataset.
        """
        dataset = load_dataset("glue", "cola")
        dataset = dataset.map(
            partial(cola_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence"],
        )
        return dataset

    @cache_dataset
    def load_sst2_dataset(self):
        """
        Load and tokenize the SST-2 dataset.

        This method loads the SST-2 dataset, tokenizes it using the provided
        tokenizer, and caches the tokenized dataset to disk.

        Returns:
            Dataset: The tokenized SST-2 dataset.
        """
        dataset = load_dataset("glue", "sst2")
        dataset = dataset.map(
            partial(cola_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence"],
        )
        return dataset

    @cache_dataset
    def load_qnli_dataset(self):
        """
        Load and tokenize the QNLI dataset.

        This method loads the QNLI dataset, tokenizes it using the provided
        tokenizer, and caches the tokenized dataset to disk.

        Returns:
            Dataset: The tokenized QNLI dataset.
        """
        dataset = load_dataset("glue", "qnli")
        dataset = dataset.map(
            partial(qnli_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["question", "sentence"],
        )
        return dataset
