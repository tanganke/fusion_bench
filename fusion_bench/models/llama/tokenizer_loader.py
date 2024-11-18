"""
Modified from Llama-Factory library.
"""

# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, TypedDict

from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from .patcher import patch_processor_, patch_tokenizer_

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer, ProcessorMixin


logger = logging.getLogger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def load_config(
    pretrained_model_name_or_path: str,
    trust_remote_code: bool = True,
    cache_dir: Optional[str] = None,
    model_revision: str = "main",
    hf_hub_token: Optional[str] = None,
):
    init_kwargs = {
        "trust_remote_code": trust_remote_code,
        "cache_dir": cache_dir,
        "revision": model_revision,
        "token": hf_hub_token,
    }
    return AutoConfig.from_pretrained(pretrained_model_name_or_path, **init_kwargs)


def load_tokenizer(
    pretrained_model_name_or_path: str,
    *,
    # general options for loading
    trust_remote_code: bool = True,
    cache_dir: Optional[str] = None,
    model_revision: str = "main",
    hf_hub_token: Optional[str] = None,
    # options for tokenizer
    use_fast_tokenizer: bool = True,
    split_special_tokens: bool = False,
    new_special_tokens: Optional[List[str]] = None,
    resize_vocab: bool = False,
    # options for processor
    image_resolution: int = 512,
    video_resolution: int = 128,
    video_fps: int = 2,
    video_maxlen: int = 64,
    return_dict: bool = False,
) -> "TokenizerModule":
    r"""
    Loads pretrained tokenizer and optionally loads processor.

    Args:
        use_fast_tokenizer (bool): Whether or not to use one of the fast tokenizer (backed by the tokenizers library).
        new_special_tokens (Optional[List[str]]): Special tokens to be added into the tokenizer. Use commas to separate multiple tokens.
        resize_vocab (bool): Whether or not to resize the tokenizer vocab and the embedding layers.
        trust_remote_code (bool): Whether or not to trust remote code. If True, scripts will be allowed to be downloaded and executed.
    """
    # load config
    init_kwargs = {
        "trust_remote_code": trust_remote_code,
        "cache_dir": cache_dir,
        "revision": model_revision,
        "token": hf_hub_token,
    }
    config: "PretrainedConfig" = AutoConfig.from_pretrained(
        pretrained_model_name_or_path, **init_kwargs
    )

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            use_fast=use_fast_tokenizer,
            split_special_tokens=split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    if new_special_tokens is not None:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=new_special_tokens),
            replace_additional_special_tokens=False,
        )
        logger.info("Add {} to special tokens.".format(",".join(new_special_tokens)))
        if num_added_tokens > 0 and not resize_vocab:
            logger.warning("New tokens have been added, but resize_vocab is False.")

    patch_tokenizer_(tokenizer)

    # Load processor for multimodal models
    try:
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path, **init_kwargs
        )
        patch_processor_(
            processor,
            config,
            tokenizer,
            image_resolution,
            video_resolution,
            video_fps,
            video_maxlen,
        )
    except Exception as e:
        logger.warning("Processor was not found: {}.".format(e))
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    if return_dict:
        return {"tokenizer": tokenizer, "processor": processor}
    else:
        if processor is not None:
            return tokenizer, processor
        else:
            return tokenizer
