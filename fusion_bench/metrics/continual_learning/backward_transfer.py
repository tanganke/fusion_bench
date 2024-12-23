from typing import Dict

import numpy as np


def compute_backward_transfer(
    acc_Ti: Dict[str, float], acc_ii: Dict[str, float]
) -> float:
    R"""
    Compute the backward transfer (BWT) of a model on a set of tasks.

    Equation:
        BWT = \frac{1}{n} \sum_{k=1}^{n} (acc_{Ti}[k] - acc_{ii}[k])

    Returns:
        float: The backward transfer of the model.
    """
    assert set(acc_ii.keys()) == set(acc_Ti.keys())
    bwt = 0
    for task_name in acc_ii:
        bwt += acc_Ti[task_name] - acc_ii[task_name]
    return bwt / len(acc_ii)
