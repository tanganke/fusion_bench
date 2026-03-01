from transformers import AutoConfig, AutoModel, AutoModelForImageTextToText

from .configuration_smile_qwen3_vl import SmileQwen3VLConfig
from .modeling_smile_qwen3_vl import (
    SmileQwen3VLForConditionalGeneration,
    SmileQwen3VLModel,
)

AutoConfig.register("smile_qwen3_vl", SmileQwen3VLConfig)
AutoModel.register(SmileQwen3VLConfig, SmileQwen3VLModel)
AutoModelForImageTextToText.register(
    SmileQwen3VLConfig, SmileQwen3VLForConditionalGeneration
)
