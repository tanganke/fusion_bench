"""
This scripts preprocess any NLP dataset into a text-to-text format.
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Union  # noqa: F401

from transformers import AutoTokenizer


def preprocess(
    tokenizer: AutoTokenizer,
    input_text: str,
    target_text: str,
    tokenizer_kwawgs: Dict[str, Any] = None,
):
    """
    standard preprocess function for dataset.
    Preprocesses input and target text data using a tokenizer object and returns a dictionary of model inputs.

    Args:
        tokenizer: An instance of a tokenizer class used to preprocess text data.
        input_text (str): A string containing the input text data to be tokenized.
        target_text (str, optional): A string containing the target text data to be tokenized. If None, no target data is returned.

    Returns:
        A dictionary of model inputs containing the tokenized input and output data along with the modified labels tensor.
    """
    if tokenizer_kwawgs is None:
        tokenizer_kwawgs = {}
    model_inputs = tokenizer(input_text, **tokenizer_kwawgs)
    if target_text is not None:
        labels = tokenizer(target_text, **tokenizer_kwawgs)
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
    return model_inputs


class DatasetPreprocessor:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        tokenizer_kwargs: Dict[str, Any] = None,
        template: Union[str, Path, Dict] = None,
    ):
        """
        Initializes an instance of the datasets_preprocess class with a tokenizer object.

        Args:
            tokenizer: An instance of a tokenizer class used to preprocess text data.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        if template is not None:
            if isinstance(template, str):
                template = template
                assert os.path.exists(
                    template
                ), f"Template file not found at {template}"
                with open(template, "r") as f:
                    self.template = json.load(f)
            elif isinstance(template, dict):
                self.template = template
            else:
                raise ValueError(
                    "Template must be a path to a json file or a dictionary"
                )
