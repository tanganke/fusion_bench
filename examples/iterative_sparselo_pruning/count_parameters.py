"""
This script is used to count the number of parameters in a model.

Usage:
    python count_parameters.py <model_path>
"""

import argparse

import torch
from transformers import AutoModelForCausalLM

import fusion_bench.models.modeling_losparse_llama
from fusion_bench.utils.parameters import count_parameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    )

    # count all parameters
    trainable_params, all_params = count_parameters(model)
    print("########################")
    print(f"Trainable parameters: {trainable_params}")
    print(f"All parameters: {all_params}")

    # count non-zero parameters
    trainable_params, all_params = count_parameters(model, non_zero_only=True)
    print("########################")
    print(f"Trainable parameters (non-zero only): {trainable_params}")
    print(f"All parameters (non-zero only): {all_params}")
