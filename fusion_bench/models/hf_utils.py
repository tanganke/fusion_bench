"""
This module contains utilities for working with Hugging Face models.
"""

import inspect
import os
import shutil
from typing import List, Optional, cast

from omegaconf import DictConfig, OmegaConf
from transformers.modeling_utils import PreTrainedModel

from fusion_bench.utils.pylogger import get_rankzero_logger

log = get_rankzero_logger(__name__)

__all__ = [
    "load_model_card_template",
    "save_pretrained_with_remote_code",
    "create_default_model_card",
]

MODEL_CARD_TEMPLATE_DIRS = [
    os.path.join(os.path.dirname(__file__), "model_card_templates")
]


def load_model_card_template(basename: str) -> str:
    """
    Load a model card template from file.

    Searches for a template file by name, first checking if the name is a direct file path,
    then searching through predefined template directories.

    Args:
        name (str): The name of the template file or a direct file path to the template.

    Returns:
        str: The contents of the template file as a string.

    Raises:
        FileNotFoundError: If the template file is not found in any of the search locations.
    """
    if os.path.exists(basename):
        with open(basename, "r") as f:
            return f.read()

    for template_dir in MODEL_CARD_TEMPLATE_DIRS:
        template_path = os.path.join(template_dir, basename)
        if os.path.exists(template_path):
            with open(template_path, "r") as f:
                return f.read()

    raise FileNotFoundError(f"Model card template '{basename}' not found.")


def try_to_yaml(config):
    if config is None:
        return None

    try:
        return OmegaConf.to_yaml(config, resolve=True, sort_keys=True)
    except Exception as e:
        log.error(f"Failed to convert config to YAML: {e}. Return `None`.")
        return None


def save_pretrained_with_remote_code(
    model: PreTrainedModel,
    auto_map: dict[str, object],
    save_directory,
    **kwargs,
):
    """
    Saves a model with custom code to a directory.

    This function facilitates saving a Hugging Face `PreTrainedModel` along with its
    associated custom code. It inspects the objects provided in the `auto_map`,
    copies their source files to the `save_directory`, and generates an `__init__.py`
    to make them importable. It also updates the model's configuration with an
    `auto_map` attribute, which allows `AutoModel.from_pretrained` to correctly
    instantiate the custom model classes when `trust_remote_code=True`.

    Args:
        model (PreTrainedModel): The model instance to be saved.
        auto_map (dict[str, object]): A dictionary mapping auto class names
            (e.g., "AutoModelForCausalLM") to the corresponding custom class objects.
        save_directory (str or os.PathLike): The directory where the model and
            custom code files will be saved.
        **kwargs: Additional keyword arguments to be passed to the
            `model.save_pretrained` method.

    Example:
        ```python
        # Assuming `model` is an instance of `SmileQwen2ForCausalLM`
        # and `SmileQwen2Config`, `SmileQwen2Model`, `SmileQwen2ForCausalLM`
        # are custom classes defined in your project.

        save_pretrained_with_remote_code(
            model,
            auto_map={
                "AutoConfig": SmileQwen2Config,
                "AutoModel": SmileQwen2Model,
                "AutoModelForCausalLM": SmileQwen2ForCausalLM,
            },
            save_directory="./my-custom-model",
        )

        # The model can then be loaded with `trust_remote_code=True`:
        # from transformers import AutoModelForCausalLM
        # loaded_model = AutoModelForCausalLM.from_pretrained(
        #     "./my-custom-model", trust_remote_code=True
        # )
        ```
    """
    auto_map_files = {}
    auto_map_strs = {}
    for key, obj in auto_map.items():
        auto_map_files[key] = inspect.getfile(obj)

    for key, obj in auto_map.items():
        auto_map_strs[key] = (
            f"{(inspect.getmodule(obj).__name__).split('.')[-1]}.{obj.__name__}"
        )

    model.config.auto_map = auto_map_strs

    # save model to `save_directory`
    model.save_pretrained(save_directory=save_directory, **kwargs)

    # copy source files to `save_directory`
    for key, file_path in auto_map_files.items():
        shutil.copy(
            src=file_path, dst=os.path.join(save_directory, os.path.basename(file_path))
        )
    # construct `__init__.py`
    init_file = os.path.join(save_directory, "__init__.py")
    with open(init_file, "w") as f:
        for key, file_name in auto_map_files.items():
            base_name = os.path.basename(file_name).split(".")[0]
            f.write(f"from .{base_name} import {auto_map[key].__name__}\n")


def create_default_model_card(
    models: Optional[list[str]] = None,
    base_model: Optional[str] = None,
    title: str = "Deep Model Fusion",
    tags: list[str] = ["fusion-bench", "merge"],
    description=None,
    algorithm_config: DictConfig = None,
    modelpool_config: DictConfig = None,
):
    from jinja2 import Template

    if models is None:
        models = []

    template: Template = Template(load_model_card_template("default.md"))
    card = template.render(
        base_model=base_model,
        models=models,
        library_name="transformers",
        title=title,
        tags=tags,
        description=description,
        algorithm_config_str=try_to_yaml(algorithm_config),
        modelpool_config_str=try_to_yaml(modelpool_config),
    )
    return card
