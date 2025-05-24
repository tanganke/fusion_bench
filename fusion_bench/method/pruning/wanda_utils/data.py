# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import os
import random
from typing import List, Optional, Tuple, cast  # noqa: F401

from torch import Tensor
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from datasets import load_dataset


# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


# Load and process wikitext2 dataset
def get_wikitext2(
    nsamples: int,
    seed: int,
    seqlen: int,
    tokenizer: PreTrainedTokenizer,
    data_path: str = "wikitext",
):
    """
    Load and preprocess the Wikitext-2 dataset for training and testing.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Length of the sequence to be used for training.
        tokenizer (PreTrainedTokenizer): Tokenizer to encode the text data.
        data_path (str, optional): Path to the dataset. Defaults to "wikitext".
    """
    # Load train and test datasets
    traindata = load_dataset(data_path, "wikitext-2-raw-v1", split="train")
    testdata = load_dataset(data_path, "wikitext-2-raw-v1", split="test")

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    # Generate samples from training set
    random.seed(seed)
    trainloader: List[Tuple[Tensor, Tensor]] = []
    for _ in tqdm(range(nsamples), desc="Generating samples"):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp: Tensor = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


# Load and process c4 dataset
def get_c4(
    nsamples: int,
    seed: int,
    seqlen: int,
    tokenizer,
    data_path: str = "allenai/c4",
    cache_dir: str = ".cache/allenai--c4",
) -> Tuple[List[Tuple[Tensor, Tensor]], TokenizerWrapper]:
    """
    Load and process the c4 dataset.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Seed for random number generation.
        seqlen (int): Length of each sequence.
        tokenizer: Tokenizer object for encoding the text.
        data_path (str, optional): Path to the c4 dataset. Defaults to "allenai/c4".

    Returns:
        tuple (Tuple[List[Tuple[Tensor, Tensor]], TokenizerWrapper]): Tuple containing the training samples and the validation dataset.
    """
    # Load train and validation datasets
    if os.path.exists(f"{cache_dir}/en/c4-train.00000-of-01024.json.gz"):
        traindata = load_dataset(
            "json",
            data_files={"train": f"{cache_dir}/en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
    else:
        traindata = load_dataset(
            data_path,
            # "allenai--c4", # https://github.com/huggingface/datasets/issues/6559
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )

    if os.path.exists(f"{cache_dir}/en/c4-validation.00000-of-00008.json.gz"):
        valdata = load_dataset(
            "json",
            data_files={
                "validation": f"{cache_dir}/en/c4-validation.00000-of-00008.json.gz",
            },
            split="validation",
        )
    else:
        valdata = load_dataset(
            data_path,
            # "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
    # Generate samples from training set
    if seed is not None:
        random.seed(seed)

    trainloader = []
    for _ in tqdm(range(nsamples), desc="Generating samples"):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


# Function to select the appropriate loader based on dataset name
def get_loaders(
    name: str, nsamples: int = 128, seed: int = 0, seqlen: int = 2048, tokenizer=None
):
    """
    Get the data loaders for the specified dataset.

    Args:
        name (str): The name of the dataset. Supported values are "wikitext2" and "c4".
        nsamples (int, optional): Number of samples to generate from the dataset. Defaults to 128.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        seqlen (int, optional): Length of the sequence to be used for training. Defaults to 2048.
        tokenizer (optional): Tokenizer to encode the text data. Defaults to None.
    """
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    raise ValueError(f"Unknown dataset: {name}")
