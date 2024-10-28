import logging
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def remove_special_tokens(tokenizer, token_list: list):
    """
    This function removes special tokens from a list of tokens. It also stops processing
    when it encounters a token with a value of -100.

    Parameters:
        tokenizer (Tokenizer): The tokenizer object used for tokenizing text.
        token_list (list): The list of tokens to be processed.

    Returns:
        list: The list of tokens after removing special tokens.
    """
    ret = []
    for token in token_list:
        if token not in tokenizer.all_special_ids and token > 0:
            ret.append(token)
        if token == -100:
            break
    return ret


def evaluate_accuracy(model, val_loader: DataLoader, tokenizer):
    """
    This function evaluates the accuracy of a language model on a validation set.

    Parameters:
        model (nn.Module): The language model to be evaluated.
        val_loader (DataLoader): The DataLoader object containing the validation data.
        tokenizer (Tokenizer): The tokenizer object used for tokenizing text.

    Returns:
        float: The accuracy of the model on the validation set.
    """
    from tqdm import tqdm

    correct = 0
    total = 0

    model = model.eval()
    for batch_idx, batch in enumerate(
        tqdm(
            val_loader, desc="Evaluate Exact Accuracy", leave=False, dynamic_ncols=True
        )
    ):
        with torch.no_grad():
            outputs = model.generate(batch["input_ids"], max_length=10)
            output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            labels = [
                remove_special_tokens(tokenizer, label_token)
                for label_token in batch["labels"]
            ]
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # compare output_text and labels
            for i, j in zip(output_text, labels):
                if i == j:
                    correct += 1
                total += 1

    # return accuracy
    return correct / total


def evaluate_spearman_rho(model, val_loader: DataLoader, tokenizer):
    """
    This function evaluates the Spearman's rank correlation coefficient (rho) between the model's predictions and the actual labels on a validation set.

    Parameters:
        model (nn.Module): The language model to be evaluated.
        val_loader (DataLoader): The DataLoader object containing the validation data.
        tokenizer (Tokenizer): The tokenizer object used for tokenizing text.

    Returns:
        float: The Spearman's rho between the model's predictions and the actual labels.
    """
    from tqdm import tqdm

    model = model.eval()
    all_preds: List[str] = []
    all_labels: List[str] = []
    for batch_idx, batch in enumerate(
        tqdm(val_loader, desc="Evaluate Spearman Rho", leave=False, dynamic_ncols=True)
    ):
        with torch.no_grad():
            outputs = model.generate(batch["input_ids"], max_length=10)
            output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            labels = [
                remove_special_tokens(tokenizer, label_token)
                for label_token in batch["labels"]
            ]
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_preds.extend(output_text)
            all_labels.extend(labels)

    # save `all_preds` and `all_labels`
    # with open("temp/all_preds.txt", "w") as f:
    #     for preds in all_preds:
    #         for pred in preds:
    #             f.write(pred + "\n")
    # with open("temp/all_labels.txt", "w") as f:
    #     for labels in all_labels:
    #         for label in labels:
    #             f.write(label + "\n")

    # calculate spearman's rho
    # 1. convert string list `all_preds` and `all_labels` to numpy array
    # 2. compute spearman's rho
    from scipy.stats import spearmanr

    def parse_flost(s: str):
        try:
            return float(s)
        except Exception:
            return 0.0

    all_preds = np.array([parse_flost(pred) for pred in all_preds])
    all_labels = np.array([parse_flost(label) for label in all_labels])
    rho = spearmanr(all_preds, all_labels)[0]
    return rho
