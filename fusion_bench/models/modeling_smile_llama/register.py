from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_smile_llama import SmileLlamaConfig
from .modeling_smile_llama import SmileLlamaForCausalLM, SmileLlamaModel

AutoConfig.register("smile_llama", SmileLlamaConfig)
AutoModel.register(SmileLlamaConfig, SmileLlamaModel)
AutoModelForCausalLM.register(SmileLlamaConfig, SmileLlamaForCausalLM)
