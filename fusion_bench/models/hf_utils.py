"""
This module contains utilities for working with Hugging Face models.
"""

import inspect
import os
import shutil
from typing import Optional, cast

from omegaconf import OmegaConf
from transformers.modeling_utils import PreTrainedModel

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.utils.pylogger import getRankZeroLogger

log = getRankZeroLogger(__name__)

__all__ = [
    "save_pretrained_with_remote_code",
    "generate_readme_head",
    "generate_readme_body",
    "generate_complete_readme",
]


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


def generate_readme_head(
    models: list[str] | BaseModelPool,
    library_name: str = "transformers",
    tags: list[str] = ["fusion-bench", "merge"],
):
    text = "---\nbase_model:\n"
    for model_name in models:
        text += f"- {model_name}\n"
    if library_name:
        text += f"library_name: {library_name}\n"
    text += "tags:\n"
    for tag in tags:
        text += f"- {tag}\n"
    text += "---\n"
    return text


def generate_readme_body(
    algorithm: BaseAlgorithm,
    models_or_modelpool: Optional[list[str] | BaseModelPool] = None,
    models: list[str] = None,
):
    text = """\
# Merge

This is a merge of pre-trained language models created using [fusion-bench](https://github.com/tanganke/fusion_bench).

"""

    if models is not None:
        text += """
## Models Merged

The following models were included in the merge:

"""
        for model_name in models:
            text += f"- {model_name}\n"
        text += "\n"

        try:
            text += f"""\
    ## Configuration

    The following YAML configuration was used to produce this model:

    ```yaml
    {OmegaConf.to_yaml(algorithm.config, resolve=True, sort_keys=True)}
    ```
    """
        except Exception as e:
            return (
                text  # If the algorithm config cannot be converted to YAML, we skip it.
            )

    if isinstance(models_or_modelpool, BaseModelPool):
        try:
            text += f"""
```yaml
{OmegaConf.to_yaml(models_or_modelpool.config, resolve=True, sort_keys=True)}
```
"""
        except Exception as e:
            pass  # If the model pool config cannot be converted to YAML, we skip it.
    return text


def generate_complete_readme(
    algorithm: BaseAlgorithm, modelpool: BaseModelPool, models: list[str]
):
    # Generate the complete README content
    text = generate_readme_head(
        [modelpool.get_model_path(m) for m in modelpool.model_names]
    )
    readme_body = generate_readme_body(
        algorithm,
        models_or_modelpool=modelpool,
        models=[modelpool.get_model_path(m) for m in modelpool.model_names],
    )
    complete_readme = text + "\n" + readme_body
    return complete_readme
