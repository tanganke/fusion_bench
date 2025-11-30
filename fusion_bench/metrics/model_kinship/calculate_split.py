import logging
from typing import Dict, List

import numpy
import torch
from tqdm import tqdm

from .utility import Metric, load_model_state_dict, quantize_8bit


def cosine_similarity(a, b):
    similarity = numpy.sqrt(numpy.dot(a, b) ** 2 / (numpy.dot(a, a) * numpy.dot(b, b)))
    return similarity


def calculate_model_kinship_split(
    model_1_name: str,
    model_2_name: str,
    model_base_name: str,
    low_precision: bool,
    metrics: List[str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:

    # Extract state dictionaries from models
    state_dict_1 = load_model_state_dict(model_1_name, device)
    state_dict_2 = load_model_state_dict(model_2_name, device)
    state_dict_base = load_model_state_dict(model_base_name, device)
    results = {}

    # Validate metrics before processing
    valid_metrics = Metric.list()
    for metric in metrics:
        try:
            if metric not in valid_metrics:
                raise ValueError(
                    f"Unsupported metric: {metric}. Valid metrics are: {', '.join(valid_metrics)}"
                )
            results[metric] = calculate_metrics_by_split(
                state_dict_1, state_dict_2, state_dict_base, low_precision, metric
            )
        except Exception as e:
            logging.error(f"Error calculating {metric}: {str(e)}")
            results[metric] = f"Error calculating {metric}: {str(e)}"

    return results


def calculate_metrics_by_split(
    state_dict_1: dict,
    state_dict_2: dict,
    state_dict_base: dict,
    low_precision: bool,
    metric: str,
) -> str:
    """
    Calculate metrics for each key and integrate results.

    Args:
        state_dict_1 (dict): State dictionary of first model
        state_dict_2 (dict): State dictionary of second model
        state_dict_base (dict): State dictionary of base model
        low_precision (bool): Whether to use 8-bit quantization
        metric (str): Metric to calculate ('pcc', 'ed', 'cs')

    Returns:
        str: Integrated metric result as formatted string
    """
    total_similarity = 0.0
    total_weight = 0.0
    split_results = {}

    # Determine the number of layers
    num_layers = state_dict_base["lm_head.weight"].shape[0]

    # Check architectures
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

    # Process each key
    for key, base_params in tqdm(
        state_dict_base.items(), desc=f"Processing {metric.upper()} by key"
    ):
        try:
            if key not in state_dict_1 or key not in state_dict_2:
                logging.warning(f"Key {key} not found in one of the models")
                continue

            # Get parameters and calculate deltas
            params_1 = state_dict_1[key][:num_layers]
            params_2 = state_dict_2[key][:num_layers]

            delta_1 = (params_1 - base_params).view(-1)
            delta_2 = (params_2 - base_params).view(-1)

            if low_precision:
                delta_1 = quantize_8bit(delta_1)
                delta_2 = quantize_8bit(delta_2)

            # Calculate weight based on parameter count
            weight = delta_1.numel()

            # Calculate metric for current key
            if metric == "pcc":
                stack = torch.stack((delta_1, delta_2), dim=0)
                split_similarity = torch.corrcoef(stack)[0, 1].item()
            elif metric == "ed":
                split_similarity = torch.dist(delta_1, delta_2).item()
            elif metric == "cs":
                split_similarity = cosine_similarity(delta_1, delta_2)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            # Skip NaN values
            if torch.isnan(torch.tensor(split_similarity)):
                logging.warning(f"Skipping key {key} due to NaN result")
                continue

            # Store valid result
            split_results[key] = split_similarity

            # Update weighted average only for valid results
            weight = delta_1.numel()
            total_similarity += split_similarity * weight
            total_weight += weight

            # Log progress for large layers
            if weight > 1000000:
                logging.info(
                    f"Layer {key}: {metric.upper()} = {split_similarity:.4f}, parameters = {weight}"
                )

            # Free memory
            del delta_1, delta_2

        except Exception as e:
            logging.error(f"Error processing key {key}: {str(e)}")
            continue

    # Calculate final weighted average
    if total_weight > 0:
        final_result = total_similarity / total_weight

        # Log summary statistics
        logging.info(f"\nSummary for {metric.upper()}:")
        logging.info(f"Total parameters: {total_weight}")

        # Log detailed results for valid splits
        logging.info(f"\nDetailed {metric.upper()} results by key:")
        for key, value in split_results.items():
            logging.info(f"{key}: {value:.4f}")

        metric_names = {
            "pcc": "Pearson Correlation Coefficient",
            "ed": "Euclidean Distance",
            "cs": "Cosine Similarity",
        }

        return f"Model Kinship based on {metric_names[metric]} (weighted average): {final_result:.4f}"
    else:
        return f"Error: No valid parameters found for {metric.upper()} calculation"
