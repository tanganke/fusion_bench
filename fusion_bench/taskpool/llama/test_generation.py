import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

import torch

from fusion_bench import BaseTaskPool
from fusion_bench.taskpool.dummy import get_model_summary
from fusion_bench.utils.devices import get_device
from fusion_bench.utils.rich_utils import print_bordered

if TYPE_CHECKING:
    from transformers import LlamaForCausalLM, PreTrainedTokenizer

    from fusion_bench.modelpool import CausalLMPool


def generate_text(
    model: "LlamaForCausalLM",
    tokenizer: "PreTrainedTokenizer",
    prompt: str,
    max_length: int = 1024,
    temperature: float = 0.01,
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
    response = generated_text[len(prompt) :]
    return {
        "generated_text": generated_text,
        "response": response,
        "num_tokens": len(outputs[0]) - len(inputs["input_ids"][0]),
    }


class LlamaTestGenerationTaskPool(BaseTaskPool):
    """
    This task pool is used to evaluate a language model on a set of prompts.
    For the purpose of debugging, it can also be used in an interactive mode.
    """

    def __init__(
        self,
        test_prompts: List[str],
        max_length: int = 1024,
        temperature: float = 0.01,
        top_p: float = 0.9,
        iterative_mode: bool = False,
        **kwargs,
    ):
        """
        Args:
            test_prompts (List[str]): A list of prompts to be used for testing the model.
            max_length (int, optional): The maximum length of the generated text. Defaults to 1024.
            temperature (float, optional): The sampling temperature for text generation. Defaults to 0.01.
            top_p (float, optional): The cumulative probability for nucleus sampling. Defaults to 0.9.
            iterative_mode (bool, optional): If True, enables interactive mode for debugging. Defaults to False.
        """
        self.test_prompts = test_prompts
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.iterative_mode = iterative_mode
        super().__init__(**kwargs)

    def evaluate(self, model: Union["LlamaForCausalLM", Any]):
        modelpool: "CausalLMPool" = self._program.modelpool
        tokenizer = modelpool.load_tokenizer()

        report = get_model_summary(model)
        if self.test_prompts is not None:
            for prompt_idx, prompt in enumerate(self.test_prompts):
                print(f"=== Generating text {prompt_idx}/{len(self.test_prompts)}")
                print(prompt)
                start_time = time.time()
                outputs = generate_text(
                    model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                print_bordered(outputs["response"])
                print("\n")

                report[f"prompt_{prompt_idx}"] = {
                    "prompt": prompt,
                    "response": outputs["response"],
                    "wall_time": time.time() - start_time,
                    "num_chars": len(outputs["response"]),
                    "num_tokens": outputs["num_tokens"],
                }

        if self.iterative_mode:
            while True:
                # Prompt for input
                # print usage instructions
                print("Enter a prompt to generate text. Type 'exit' to exit the loop.")
                prompt = input("Enter a prompt, or type 'exit' to quit: ")
                if prompt == "exit":
                    break
                start_time = time.time()
                outputs = generate_text(
                    model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                print_bordered(outputs["response"])
                print("\n")

                report[f"iterative_{len(report)}"] = {
                    "prompt": prompt,
                    "response": outputs["response"],
                    "wall_time": time.time() - start_time,
                    "num_chars": len(outputs["response"]),
                    "num_tokens": outputs["num_tokens"],
                }

        return report
