import os
from typing import Tuple

from huggingface_hub import hf_hub_download
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer
from safetensors.torch import load_file
from transformers import CLIPVisionModel
from transformers.models.clip.modeling_clip import CLIPVisionTransformer

from .linearized_model_utils import LinearizedModelWraper


def get_file_path(peft_name, filename):
    if os.path.isdir(peft_name):
        # If peft_name is a local directory path
        return os.path.join(peft_name, filename)
    else:
        # If peft_name is a Hugging Face model name
        return hf_hub_download(peft_name, filename)


def _get_submodules(model, key) -> Tuple:
    """
    Retrieves the parent module, target module, and target module name for a given key in a PyTorch model.
    """
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def linearize_lora_model_(model):
    """
    Linearizes the LoraLayer modules in a PyTorch model according to the PETA paper.
    """
    for key, module in model.named_modules():
        # if isinstance(module, LoraLayer) and isinstance(module, nn.Linear):
        if isinstance(module, LoraLayer):
            # print("L-LoRA MODULE : ", module)
            parent, target, target_name = _get_submodules(model, key)
            setattr(parent, target_name, LinearizedModelWraper(target))
            # print("Linearized Lora Layer")
    return model


def load_fft_vision_model_hf(model_name: str) -> CLIPVisionTransformer:
    return CLIPVisionModel.from_pretrained(model_name).vision_model


def load_lora_vision_model_hf(base_model_name: str, peft_name: str):
    model = CLIPVisionModel.from_pretrained(base_model_name).vision_model
    return PeftModel.from_pretrained(model, peft_name, is_trainable=True)


def load_l_lora_vision_model_hf(base_model_name: str, peft_name: str):
    """
    Load a linearized L-LoRA model from a base model and a Peft model (HuggingFace).
    """
    base_model = CLIPVisionModel.from_pretrained(base_model_name).vision_model
    peft_config = LoraConfig.from_pretrained(peft_name)
    peft_config.inference_mode = False  # This is important, make the model trainable
    model = get_peft_model(base_model, peft_config)
    linearize_lora_model_(model)
    for filename in ["linearized_adapter_model.safetensors"]:
        path = get_file_path(peft_name, filename)
        state_dict = load_file(path)
        for name, param in state_dict.items():
            model.get_parameter(name).data = param

    return model
