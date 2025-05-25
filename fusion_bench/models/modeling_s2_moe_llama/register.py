from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_s2_moe_llama import S2MoELlamaConfig
from .modeling_s2_moe_llama import S2MoELlamaForCausalLM, S2MoELlamaModel

AutoConfig.register("s2_moe_llama", S2MoELlamaConfig)
AutoModel.register(S2MoELlamaConfig, S2MoELlamaModel)
AutoModelForCausalLM.register(S2MoELlamaConfig, S2MoELlamaForCausalLM)
