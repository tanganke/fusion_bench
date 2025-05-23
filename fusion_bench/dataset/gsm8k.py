import re
from typing import Literal

import datasets
from datasets import load_dataset


def load_gsm8k_question_label_data(
    dataset_name: Literal["train", "test", "train_socratic", "test_socratic"],
):
    R"""
    Load the GSM8K dataset and extract questions and labels.

    An example in the dataset:

    {'question': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?',
     'answer': 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72'}

    Args:
        dataset_name (Literal["train", "test", "train_socratic", "test_socratic"]): The name of the dataset to load.

    Returns:
        questions (List[str]): List of questions.
        labels (List[float]): List of labels. For example, the label for the above example is `72.0`.
    """
    # Load the dataset
    if dataset_name in ["train", "test"]:
        dataset = load_dataset("openai/gsm8k", "main", split=dataset_name)
    elif dataset_name in ["train_socratic", "test_socratic"]:
        dataset = load_dataset(
            "openai/gsm8k", "socratic", split=dataset_name.split("_")[0]
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # Extract the questions and labels
    questions, labels = [], []
    pattern = r"####\s*(\S+)"
    for i, sample in enumerate(dataset):
        matches = re.findall(pattern, sample["answer"])
        labels.append(float(matches[0].replace(",", "")))
        questions.append(sample["question"])

    return questions, labels


def load_gsm8k_question_label_dataset(
    dataset_name: Literal["train", "test", "train_socratic", "test_socratic"],
):
    """
    Load the GSM8K dataset and return it as a Hugging Face Dataset object.

    Args:
        dataset_name (Literal["train", "test", "train_socratic", "test_socratic"]): The name of the dataset to load.

    Returns:
        datasets.Dataset: The loaded dataset with questions and labels.
    """
    question, labels = load_gsm8k_question_label_data(dataset_name)
    return datasets.Dataset.from_dict({"question": question, "label": labels})
