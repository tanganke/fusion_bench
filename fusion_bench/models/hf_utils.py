"""
This module contains utilities for working with Hugging Face models.
"""

import inspect
import os
import shutil
from typing import cast

from transformers.modeling_utils import PreTrainedModel

from fusion_bench.utils.pylogger import getRankZeroLogger

log = getRankZeroLogger(__name__)

__all__ = ["save_pretrained_with_remote_code"]


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
