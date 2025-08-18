from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from .configuration_smile_gemma2 import SmileGemma2Config
from .modeling_smile_gemma2 import (
    SmileGemma2ForCausalLM,
    SmileGemma2ForSequenceClassification,
    SmileGemma2ForTokenClassification,
    SmileGemma2Model,
    SmileGemma2PreTrainedModel,
)

AutoConfig.register("smile_gemma2", SmileGemma2Config)
AutoModel.register(SmileGemma2Config, SmileGemma2Model)
AutoModelForCausalLM.register(SmileGemma2Config, SmileGemma2ForCausalLM)
AutoModelForSequenceClassification.register(
    SmileGemma2Config, SmileGemma2ForSequenceClassification
)
AutoModelForTokenClassification.register(
    SmileGemma2Config, SmileGemma2ForTokenClassification
)
