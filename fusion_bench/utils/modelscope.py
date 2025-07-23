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
    # If it's a local file or directory, return as is
    if os.path.exists(repo_id):
        return repo_id
    # If it's a HuggingFace Hub model id, download snapshot
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
