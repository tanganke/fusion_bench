import logging
from enum import Enum
from typing import List

import click
import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
)


class Metric(str, Enum):
    """Enumeration of supported metrics"""

    PCC = "pcc"
    ED = "ed"
    CS = "cs"

    @classmethod
    def list(cls) -> List[str]:
        """Return list of supported metric values"""
        return [metric.value for metric in cls]


def get_config(model: str, trust_remote_code: bool = False) -> PretrainedConfig:
    """
    Fetch the configuration of a pretrained model from HuggingFace.

    Args:
        model (str): The name or path of the model to load configuration for.
        trust_remote_code (bool, optional): Whether to trust remote code during loading.
                                            Defaults to False.

    Returns:
        PretrainedConfig: The configuration object of the specified model.
    """
    # Fetch the configuration from HuggingFace's model hub.
    config = AutoConfig.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,  # Whether to allow remote code execution.
    )
    return config


def validate_models(model_1: str, model_2: str, base_model: str) -> None:
    """
    Validate model names to ensure they are different and exist.

    Args:
        model_1: Name of the first model
        model_2: Name of the second model
        base_model: Name of the base model

    Raises:
        click.BadParameter: If validation fails
    """
    if model_1 == model_2 or model_1 == base_model or model_2 == base_model:
        raise click.BadParameter("All model names must be different")


def quantize_8bit(x: torch.Tensor) -> torch.Tensor:
    # Get absolute min and max values
    abs_max = torch.max(torch.abs(x))

    # Scale to [-127, 127] range for 8-bit signed integers
    # Using 127 instead of 128 to keep zero exactly representable
    scaled = 127 * (x / abs_max)

    # Round to nearest integer
    quantized = torch.round(scaled)

    # Clamp values to ensure they stay in valid range
    quantized = torch.clamp(quantized, -127, 127)

    return quantized


def load_model_state_dict(model_name: str, device: str) -> dict:
    """
    Load a model and return its state dictionary.

    Args:
        model_name (str): Name or path of the model to load
        device (str): Device to load the model on ('cuda' or 'cpu')

    Returns:
        dict: State dictionary of the loaded model
    """
    logging.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    state_dict = model.state_dict()
    del model  # Free memory
    return state_dict


def extract_delta_parameters(
    model_1_name: str,
    model_2_name: str,
    model_base_name: str,
    low_precision: bool,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract the delta parameters (weight differences) between two models
    relative to a base model.

    Args:
        model_1_name (str): Name or path of the first model.
        model_2_name (str): Name or path of the second model.
        model_base_name (str): Name or path of the base model for comparison.
        low_precision (bool): Whether to use low precision weights

    Returns:
        (torch.Tensor, torch.Tensor): Delta parameters of model_1 and model_2 relative to base model.
    """

    # Extract state dictionaries from models
    state_dict_1 = load_model_state_dict(model_1_name, device)
    state_dict_2 = load_model_state_dict(model_2_name, device)
    state_dict_base = load_model_state_dict(model_base_name, device)

    # Determine the number of layers
    num_layers = state_dict_base["lm_head.weight"].shape[0]

    # Check if model architectures match, log a warning if not
    if (
        state_dict_1["lm_head.weight"].shape[0]
        != state_dict_2["lm_head.weight"].shape[0]
    ):
        shape_1 = state_dict_1["lm_head.weight"].shape
        shape_2 = state_dict_2["lm_head.weight"].shape
        logging.warning(
            f"Warning: Model architectures do not match. "
            f"Using sub weight space instead.\n"
            f"Vocab sizes in model 1: {shape_1[0]}, "
            f"Vocab sizes in model 2: {shape_2[0]}"
        )

    # Initialize lists to store delta parameters for both models
    d_vector_1, d_vector_2 = [], []

    # Iterate over keys in the base model's state dictionary with tqdm
    for key, base_params in tqdm(
        state_dict_base.items(), desc="Processing keys", unit="key"
    ):
        # Only proceed if key exists in both models
        try:
            if key not in state_dict_1 or key not in state_dict_2:
                logging.warning(f"Key {key} not found in one of the models")
                continue
        except Exception as e:
            logging.error(f"Error processing key {key}: {str(e)}")

        # Get the parameters for each model (truncate to num_layers for consistency)
        params_1 = state_dict_1[key][:num_layers]
        params_2 = state_dict_2[key][:num_layers]

        # Compute the deltas relative to the base model
        delta_1 = (params_1 - base_params).view(-1)
        delta_2 = (params_2 - base_params).view(-1)

        # Accumulate deltas
        d_vector_1.append(delta_1)
        d_vector_2.append(delta_2)

    # Clear memory
    del state_dict_1, state_dict_2, state_dict_base

    logging.info("Concatenating delta vectors...")

    d_vector_1 = torch.cat(d_vector_1)
    d_vector_2 = torch.cat(d_vector_2)

    if low_precision:
        logging.info("Quantizing delta vectors to 8-bit precision...")
        d_vector_1 = quantize_8bit(d_vector_1)
        d_vector_2 = quantize_8bit(d_vector_2)
        logging.info("Quantization complete")

    return d_vector_1, d_vector_2
