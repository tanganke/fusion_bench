from typing import List, Union

import numpy as np


def layer_load_balance_score(
    number_of_tokens_dispatched: Union[List[int], np.ndarray],
    number_of_experts: int,
) -> float:
    """
    Calculate the load balance score for one layer of the MoE model.

    Args:
        number_of_tokens_dispatched: List[int]
        number_of_experts: int

    Returns:
        float: The load balance score
    """
    if len(number_of_tokens_dispatched) != number_of_experts:
        raise ValueError(
            f"The number of tokens dispatched ({len(number_of_tokens_dispatched)}) must match the number of experts ({number_of_experts})"
        )

    number_of_tokens_dispatched = np.array(number_of_tokens_dispatched)
    mu = number_of_tokens_dispatched.mean()
    sigma = np.sqrt(((number_of_tokens_dispatched - mu) ** 2).mean())
    return sigma / mu


def model_load_balance_score(layer_load_balance_scores: List[float]) -> float:
    """
    Calculate the load balance score for the whole model.

    Args:
        layer_load_balance_scores: List[float]

    Returns:
        float: The load balance score
    """
    return np.array(layer_load_balance_scores).mean()
