import sys
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from fusion_bench.utils.lazy_imports import LazyImporter

# from .AutoModelForSeq2SeqLM import AutoModelForSeq2SeqLMPool
# from .huggingface_clip_vision import HuggingFaceClipVisionPool
# from .huggingface_gpt2_classification import HuggingFaceGPT2ClassificationPool
# from .PeftModelForSeq2SeqLM import PeftModelForSeq2SeqLMPool

_import_structure = {
    "base_pool": ["BaseModelPool"],
    "clip_vision": ["CLIPVisionModelPool"],
}


# class ModelPoolFactory:
#     _modelpool = {
#         "NYUv2ModelPool": ".nyuv2_modelpool.NYUv2ModelPool",
#         "huggingface_clip_vision": HuggingFaceClipVisionPool,
#         "HF_GPT2ForSequenceClassification": HuggingFaceGPT2ClassificationPool,
#         "AutoModelPool": ".huggingface_automodel.AutoModelPool",
#         # CausualLM
#         "AutoModelForCausalLMPool": ".huggingface_llm.AutoModelForCausalLMPool",
#         "LLamaForCausalLMPool": ".huggingface_llm.LLamaForCausalLMPool",
#         "MistralForCausalLMPool": ".huggingface_llm.MistralForCausalLMPool",
#         # Seq2SeqLM
#         "AutoModelForSeq2SeqLMPool": AutoModelForSeq2SeqLMPool,
#         "PeftModelForSeq2SeqLMPool": PeftModelForSeq2SeqLMPool,
#     }

#     @staticmethod
#     def create_modelpool(modelpool_config: DictConfig):
#         from fusion_bench.utils import import_object

#         modelpool_type = modelpool_config.get("type")
#         if modelpool_type is None:
#             raise ValueError("Model pool type not specified")

#         if modelpool_type not in ModelPoolFactory._modelpool:
#             raise ValueError(
#                 f"Unknown model pool: {modelpool_type}, available model pools: {ModelPoolFactory._modelpool.keys()}. You can register a new model pool using `ModelPoolFactory.register_modelpool()` method."
#             )
#         modelpool_cls = ModelPoolFactory._modelpool[modelpool_type]
#         if isinstance(modelpool_cls, str):
#             if modelpool_cls.startswith("."):
#                 modelpool_cls = f"fusion_bench.modelpool.{modelpool_cls[1:]}"
#             modelpool_cls = import_object(modelpool_cls)
#         return modelpool_cls(modelpool_config)

#     @staticmethod
#     def register_modelpool(name: str, modelpool_cls):
#         ModelPoolFactory._modelpool[name] = modelpool_cls

#     @classmethod
#     def available_modelpools(cls):
#         return list(cls._modelpool.keys())


# def load_modelpool_from_config(modelpool_config: DictConfig):
#     """
#     Loads a model pool based on the provided configuration.

#     The function checks the 'type' attribute of the configuration and returns an instance of the corresponding model pool.
#     If the 'type' attribute is not found or does not match any known model pool types, a ValueError is raised.

#     Args:
#         modelpool_config (DictConfig): The configuration for the model pool. Must contain a 'type' attribute that specifies the type of the model pool.

#     Returns:
#         An instance of the specified model pool.

#     Raises:
#         ValueError: If 'type' attribute is not found in the configuration or does not match any known model pool types.
#     """
#     from hydra.utils import instantiate

#     return instantiate(modelpool_config, _recursive_=False)


if TYPE_CHECKING:
    from .base_pool import BaseModelPool
    from .clip_vision import CLIPVisionModelPool

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
    )
