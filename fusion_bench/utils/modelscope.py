import os
from typing import Literal, Optional

from datasets import load_dataset as datasets_load_dataset
from fusion_bench.utils import validate_and_suggest_corrections

try:
    from modelscope import snapshot_download as modelscope_snapshot_download
except ImportError:

    def modelscope_snapshot_download(*args, **kwargs):
        raise ImportError(
            "ModelScope is not installed. Please install it using `pip install modelscope` to use ModelScope models."
        )


try:
    from huggingface_hub import snapshot_download as huggingface_snapshot_download
except ImportError:

    def huggingface_snapshot_download(*args, **kwargs):
        raise ImportError(
            "Hugging Face Hub is not installed. Please install it using `pip install huggingface_hub` to use Hugging Face models."
        )


__all__ = [
    "load_dataset",
    "resolve_repo_path",
]

AVAILABLE_PLATFORMS = ["hf", "huggingface", "modelscope"]


def load_dataset(
    name: str,
    split: str = "train",
    platform: Literal["hf", "huggingface", "modelscope"] = "hf",
):
    """
    Load a dataset from Hugging Face or ModelScope.

    Args:
        platform (Literal['hf', 'modelscope']): The platform to load the dataset from.
        name (str): The name of the dataset.
        split (str): The split of the dataset to load (default is "train").

    Returns:
        Dataset: The loaded dataset.
    """
    validate_and_suggest_corrections(platform, AVAILABLE_PLATFORMS)
    if platform == "hf" or platform == "huggingface":
        return datasets_load_dataset(name, split=split)
    elif platform == "modelscope":
        dataset_dir = modelscope_snapshot_download(name, repo_type="dataset")
        return datasets_load_dataset(dataset_dir, split=split)
    else:
        raise ValueError(
            f"Unsupported platform: {platform}. Supported platforms are 'hf', 'huggingface', and 'modelscope'."
        )


def resolve_repo_path(
    repo_id: str,
    repo_type: Optional[str] = "model",
    platform: Literal["hf", "huggingface", "modelscope"] = "hf",
    **kwargs,
):
    """
    Resolve and download a repository from various platforms to a local path.

    This function handles multiple repository sources including local paths, Hugging Face,
    and ModelScope. It automatically downloads remote repositories to local cache and
    returns the local path for further use.

    Args:
        repo_id (str): Repository identifier. Can be:
            - Local file/directory path (returned as-is if exists)
            - Hugging Face model/dataset ID (e.g., "bert-base-uncased")
            - ModelScope model/dataset ID
            - URL-prefixed ID (e.g., "hf://model-name", "modelscope://model-name").
              The prefix will override the platform argument.
        repo_type (str, optional): Type of repository to download. Defaults to "model".
            Common values include "model" and "dataset".
        platform (Literal["hf", "huggingface", "modelscope"], optional):
            Platform to download from. Defaults to "hf". Options:
            - "hf" or "huggingface": Hugging Face Hub
            - "modelscope": ModelScope platform
        **kwargs: Additional arguments passed to the underlying download functions.

    Returns:
        str: Local path to the repository (either existing local path or downloaded cache path).

    Raises:
        FileNotFoundError: If the repository cannot be found or downloaded from any platform.
        ValueError: If an unsupported platform is specified.
        ImportError: If required dependencies for the specified platform are not installed.

    Examples:
        >>> # Local path (returned as-is)
        >>> resolve_repo_path("/path/to/local/model")
        "/path/to/local/model"

        >>> # Hugging Face model
        >>> resolve_repo_path("bert-base-uncased")
        "/home/user/.cache/huggingface/hub/models--bert-base-uncased/..."

        >>> # ModelScope model with explicit platform
        >>> resolve_repo_path("damo/nlp_bert_backbone_base_std", platform="modelscope")
        "/home/user/.cache/modelscope/hub/damo/nlp_bert_backbone_base_std/..."

        >>> # URL-prefixed repository ID
        >>> resolve_repo_path("hf://microsoft/DialoGPT-medium")
        "/home/user/.cache/huggingface/hub/models--microsoft--DialoGPT-medium/..."
    """
    # If it's a HuggingFace Hub model id, download snapshot
    if repo_id.startswith("hf://") or repo_id.startswith("huggingface://"):
        repo_id = repo_id.replace("hf://", "").replace("huggingface://", "")
        platform = "hf"
    # If it's a ModelScope model id, download snapshot
    elif repo_id.startswith("modelscope://"):
        repo_id = repo_id.replace("modelscope://", "")
        platform = "modelscope"

    # If it's a local file or directory, return as is
    if os.path.exists(repo_id):
        return repo_id

    try:
        validate_and_suggest_corrections(platform, AVAILABLE_PLATFORMS)
        # This will download the model to the cache and return the local path
        if platform in ["hf", "huggingface"]:
            local_path = huggingface_snapshot_download(
                repo_id=repo_id, repo_type=repo_type, **kwargs
            )
        elif platform == "modelscope":
            local_path = modelscope_snapshot_download(
                repo_id=repo_id, repo_type=repo_type, **kwargs
            )
        else:
            raise ValueError(
                f"Unsupported platform: {platform}. Supported platforms are 'hf', 'huggingface', and 'modelscope'."
            )
        return local_path
    except Exception as e:
        raise FileNotFoundError(f"Could not resolve checkpoint: {repo_id}. Error: {e}")
