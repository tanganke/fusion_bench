import logging
from typing import List

import numpy
import torch

from .utility import Metric


def cosine_similarity(a, b):
    similarity = numpy.sqrt(numpy.dot(a, b) ** 2 / (numpy.dot(a, a) * numpy.dot(b, b)))
    return similarity


def calculate_model_kinship(
    delta1: numpy.ndarray, delta2: numpy.ndarray, metrics: List[str]
) -> dict:
    """
    Calculate model kinship using specified metrics.

    Args:
        delta1: Delta parameters for first model
        delta2: Delta parameters for second model
        metrics: List of metrics to calculate

    Returns:
        dict: Dictionary of metric names and their calculated values
    """
    results = {}
    for metric in metrics:
        try:
            if metric not in Metric.list():
                raise ValueError(f"Unsupported metric: {metric}")
            results[metric] = calculate_metric(delta1, delta2, metric)
        except Exception as e:
            results[metric] = f"Error calculating {metric}: {str(e)}"
    return results


def calculate_metric(
    d_vector_1: torch.Tensor, d_vector_2: torch.Tensor, metric: str
) -> str:
    """
    Calculate the specified metric between two delta vectors.

    Args:
        d_vector_1 (torch.Tensor): Delta parameters for model 1.
        d_vector_2 (torch.Tensor): Delta parameters for model 2.
        metric (str): The metric to calculate ('pcc', 'ed', 'cs').

    Returns:
        str: A formatted string with the result of the chosen metric.
    """
    logging.info(f"Starting calculation of {metric.upper()} metric...")

    # Pearson Correlation Coefficient (PCC)
    if metric == "pcc":
        # Stack the two vectors and calculate the Pearson correlation coefficient
        stack = torch.stack((d_vector_1, d_vector_2), dim=0)
        pcc = torch.corrcoef(stack)[0, 1].item()
        return f"Model Kinship based on Pearson Correlation Coefficient: {pcc}"

    # Euclidean Distance (ED)
    elif metric == "ed":
        # Compute the Euclidean distance between the vectors
        distance = torch.dist(d_vector_1, d_vector_2).item()
        return f"Model Kinship based on Euclidean Distance: {distance}"

    # Cosine Similarity (CS)
    elif metric == "cs":
        # Compute cosine similarity
        cs = cosine_similarity(d_vector_1, d_vector_2)
        return f"Model Kinship based on Cosine Similarity: {cs}"

    # If metric is not recognized
    else:
        return "Invalid metric specified."
