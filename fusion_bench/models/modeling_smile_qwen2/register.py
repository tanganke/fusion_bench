from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_smile_qwen2 import SmileQwen2Config
from .modeling_smile_qwen2 import (
    SmileQwen2ForCausalLM,
    SmileQwen2Model,
)

AutoConfig.register("smile_qwen2", SmileQwen2Config)
AutoModel.register(SmileQwen2Config, SmileQwen2Model)
AutoModelForCausalLM.register(SmileQwen2Config, SmileQwen2ForCausalLM)
