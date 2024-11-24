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
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import torch
from peft import PeftModel
from transformers import PreTrainedTokenizerBase

from .model_utils.visual import (
    get_image_seqlen,
    get_patch_size,
    get_vision_feature_select_strategy,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer, ProcessorMixin


logger = logging.getLogger(__name__)


def patch_tokenizer_(tokenizer: "PreTrainedTokenizer") -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)


def patch_processor_(
    processor: "ProcessorMixin",
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    image_resolution: int = 512,
    video_resolution: int = 128,
    video_fps: int = 2,
    video_maxlen: int = 64,
) -> None:
    """
    Patch processor with additional attributes.

    Args:
        processor (ProcessorMixin): ProcessorMixin instance.
        config (PretrainedConfig): PretrainedConfig instance.
        tokenizer (PreTrainedTokenizer): PreTrainedTokenizer instance.
        image_resolution (int): Image resolution. Keeps the height or width of image below this resolution.
        video_resolution (int): Video resolution. Keeps the height or width of video below this resolution.
        video_fps (int): The number of frames to sample per second for video inputs.
        video_maxlen (int): The maximum number of frames to sample from video inputs.
    """
    setattr(processor, "tokenizer", tokenizer)
    setattr(processor, "image_seqlen", get_image_seqlen(config))
    setattr(processor, "image_resolution", image_resolution)
    setattr(processor, "patch_size", get_patch_size(config))
    setattr(processor, "video_resolution", video_resolution)
    setattr(processor, "video_fps", video_fps)
    setattr(processor, "video_maxlen", video_maxlen)
    setattr(
        processor,
        "vision_feature_select_strategy",
        get_vision_feature_select_strategy(config),
    )
