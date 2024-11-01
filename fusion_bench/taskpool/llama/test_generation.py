from typing import Any, Dict, List, Union, cast

import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizer

from fusion_bench import BaseTaskPool
from fusion_bench.utils.devices import get_device


def generate_text(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizer,
    prompt,
    max_length=1024,
    temperature=0.01,
    top_p=0.9,
    device: torch.device = None,
):
    """
    Generate text using the loaded model.

    Args:
        model: The loaded language model
        tokenizer: The loaded tokenizer
        prompt (str): Input prompt text
        max_length (int): Maximum length of generated sequence
        temperature (float): Controls randomness (higher = more random)
        top_p (float): Nucleus sampling parameter

    Returns:
        str: Generated text
    """
    if device is None:
        device = get_device(model)

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move to GPU if available
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )

    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return generated_text


class LlamaTestGenerationTaskPool(BaseTaskPool):

    def __init__(self, test_prompts: List[str], **kwargs):
        self.test_prompts = test_prompts
        super().__init__(**kwargs)

    def evaluate(self, model: Union[LlamaForCausalLM, Any]):
        report = {}
        