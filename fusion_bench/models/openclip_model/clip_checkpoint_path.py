import logging
from pathlib import Path

log = logging.getLogger(__name__)


def get_checkpoint_dir(cache_dir: str):
    if cache_dir is None:
        cache_dir = 'downloads'
    return Path(cache_dir) / "task_vectors_checkpoints"


MODELS = ["ViT-B-16", "ViT-B-32", "ViT-L-14"]
DATASETS = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]


def pretrained_model_path(model_name: str, cache_dir=None) -> Path:
    """
    This function generates the path for the pretrained model.

    Parameters:
        model_name (str): The name of the pretrained model.

    Returns:
        Path: The path of the pretrained model.
    """
    checkpoint_dir = get_checkpoint_dir(cache_dir)
    if model_name not in MODELS:
        log.warning(f"Unknown model {model_name}")
    path = checkpoint_dir / model_name / "zeroshot.pt"
    assert path.is_file(), f"Pretrained model not found at {path}"
    return path


def finetuned_model_path(
    model_name: str, dataset_name: str, cache_dir: str = None
) -> Path:
    """
    This function generates the path for the fine-tuned model.

    Parameters:
        model_name (str): The name of the model.
        dataset_name (str): The name of the dataset.

    Returns:
        Path: the path of the fine-tuned model.
    """
    checkpoint_dir = get_checkpoint_dir(cache_dir)
    if model_name not in MODELS:
        log.warning(f"Unknown model {model_name}")
    if dataset_name not in DATASETS:
        log.warning(f"Unknown dataset {dataset_name}")
    path = checkpoint_dir / model_name / dataset_name / "finetuned.pt"
    assert path.is_file(), f"Finetuned model not found at {path}"
    return path
