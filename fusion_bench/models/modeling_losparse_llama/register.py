from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_losparse_llama import LoSparseLlamaConfig
from .modeling_losparse_llama import LoSparseLlamaForCausalLM, LoSparseLlamaModel

AutoConfig.register("losparse_llama", LoSparseLlamaConfig)
AutoModel.register(LoSparseLlamaConfig, LoSparseLlamaModel)
AutoModelForCausalLM.register(LoSparseLlamaConfig, LoSparseLlamaForCausalLM)
