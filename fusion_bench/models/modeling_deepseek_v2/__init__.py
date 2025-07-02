"""
This is a direct copy of the DeepSeek-V2-Lite model from HuggingFace https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/tree/main
"""

from .configuration_deepseek import DeepseekV2Config
from .modeling_deepseek import (
    DeepseekV2ForCausalLM,
    DeepseekV2ForSequenceClassification,
    DeepseekV2MLP,
    DeepseekV2Model,
    DeepseekV2MoE,
    DeepseekV2DecoderLayer,
)
from .modeling_deepseek import MoEGate as DeepseekV2MoEGate
from .tokenization_deepseek_fast import DeepseekTokenizerFast
