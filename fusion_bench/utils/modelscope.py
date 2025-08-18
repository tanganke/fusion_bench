import os
from typing import Literal, Optional

from datasets import load_dataset as datasets_load_dataset

from fusion_bench.utils import validate_and_suggest_corrections

try:
    from modelscope import dataset_file_download as modelscope_dataset_file_download
    from modelscope import model_file_download as modelscope_model_file_download
    from modelscope import snapshot_download as modelscope_snapshot_download

except ImportError:

    def _raise_modelscope_not_installed_error(*args, **kwargs):
        raise ImportError(
            "ModelScope is not installed. Please install it using `pip install modelscope` to use ModelScope models."
        )

    modelscope_snapshot_download = _raise_modelscope_not_installed_error
    modelscope_model_file_download = _raise_modelscope_not_installed_error
    modelscope_dataset_file_download = _raise_modelscope_not_installed_error

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub import snapshot_download as huggingface_snapshot_download
except ImportError:

    def _raise_huggingface_not_installed_error(*args, **kwargs):
        raise ImportError(
            "Hugging Face Hub is not installed. Please install it using `pip install huggingface_hub` to use Hugging Face models."
        )

    huggingface_snapshot_download = _raise_huggingface_not_installed_error
    hf_hub_download = _raise_huggingface_not_installed_error

__all__ = [
    "load_dataset",
    "resolve_repo_path",
]

AVAILABLE_PLATFORMS = ["hf", "huggingface", "modelscope"]


def _raise_unknown_platform_error(platform: str):
    raise ValueError(
        f"Unsupported platform: {platform}. Supported platforms are 'hf', 'huggingface', and 'modelscope'."
    )


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
        _raise_unknown_platform_error(platform)


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
            _raise_unknown_platform_error(platform)
        return local_path
    except Exception as e:
        raise FileNotFoundError(f"Could not resolve checkpoint: {repo_id}. Error: {e}")


def resolve_file_path(
    repo_id: str,
    filename: str,
    repo_type: Literal["model", "dataset"] = "model",
    platform: Literal["hf", "huggingface", "modelscope"] = "hf",
    **kwargs,
) -> str:
    """
    Resolve and download a specific file from a repository across multiple platforms.

    This function downloads a specific file from repositories hosted on various platforms
    including local paths, Hugging Face Hub, and ModelScope. It handles platform-specific
    URL prefixes and automatically determines the appropriate download method.

    Args:
        repo_id (str): Repository identifier. Can be:
            - Local directory path (file will be joined with this path if it exists)
            - Hugging Face model/dataset ID (e.g., "bert-base-uncased")
            - ModelScope model/dataset ID
            - URL-prefixed ID (e.g., "hf://model-name", "modelscope://model-name").
              The prefix will override the platform argument.
        filename (str): The specific file to download from the repository.
        repo_type (Literal["model", "dataset"], optional): Type of repository.
            Defaults to "model". Used for ModelScope platform to determine the
            correct download function.
        platform (Literal["hf", "huggingface", "modelscope"], optional):
            Platform to download from. Defaults to "hf". Options:
            - "hf" or "huggingface": Hugging Face Hub
            - "modelscope": ModelScope platform
        **kwargs: Additional arguments passed to the underlying download functions
            (e.g., cache_dir, force_download, use_auth_token).

    Returns:
        str: Local path to the downloaded file.

    Raises:
        ValueError: If an unsupported repo_type is specified for ModelScope platform.
        ImportError: If required dependencies for the specified platform are not installed.
        FileNotFoundError: If the file cannot be found or downloaded.

    Examples:
        >>> # Download config.json from a Hugging Face model
        >>> resolve_file_path("bert-base-uncased", "config.json")
        "/home/user/.cache/huggingface/hub/models--bert-base-uncased/.../config.json"

        >>> # Download from ModelScope
        >>> resolve_file_path(
        ...     "damo/nlp_bert_backbone_base_std",
        ...     "pytorch_model.bin",
        ...     platform="modelscope"
        ... )
        "/home/user/.cache/modelscope/hub/.../pytorch_model.bin"

        >>> # Local file path
        >>> resolve_file_path("/path/to/local/model", "config.json")
        "/path/to/local/model/config.json"

        >>> # URL-prefixed repository
        >>> resolve_file_path("hf://microsoft/DialoGPT-medium", "config.json")
        "/home/user/.cache/huggingface/hub/.../config.json"

        >>> # Download dataset file from ModelScope
        >>> resolve_file_path(
        ...     "DAMO_NLP/jd",
        ...     "train.json",
        ...     repo_type="dataset",
        ...     platform="modelscope"
        ... )
        "/home/user/.cache/modelscope/datasets/.../train.json"
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
        return os.path.join(repo_id, filename)

    if platform in ["hf", "huggingface"]:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            **kwargs,
        )
    elif platform == "modelscope":
        if repo_type == "model":
            return modelscope_model_file_download(
                model_id=repo_id, file_path=filename, **kwargs
            )
        elif repo_type == "dataset":
            return modelscope_dataset_file_download(
                dataset_id=repo_id, file_path=filename, **kwargs
            )
        else:
            raise ValueError(
                f"Unsupported repo_type: {repo_type}. Supported types are 'model' and 'dataset'."
            )
    else:
        _raise_unknown_platform_error(platform)
