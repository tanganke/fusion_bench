"""
Merging loss defined in the paper :
"Y. Cao et al., “An Empirical Study and Theoretical Explanation on Task-Level Model-Merging Collapse,”
Mar. 10, 2026, arXiv: arXiv:2603.09463. doi: 10.48550/arXiv.2603.09463.
"""

from typing import Dict, List, Literal, Optional, Union


def _merging_loss_scalar(merged: float, individual: float) -> float:
    """Compute merging loss for a single scalar performance pair."""
    if individual == 0.0:
        raise ValueError("individual_performance is zero; merging loss is undefined.")
    return (merged / individual - 1.0) * 100.0


def compute_merging_loss(
    merged_performance: Union[float, Dict[str, float]],
    individual_performance: Union[float, Dict[str, float]],
    task_names: Optional[List[str]] = None,
    reduction: Optional[Literal["mean"]] = None,
) -> Union[float, Dict[str, float]]:
    """Compute the merging loss for one or more tasks.

    The merging loss quantifies merging collapse — the degradation in performance
    after merging multiple fine-tuned models. It is defined as:

    .. math::
        L(T_i) = \\left(\\frac{P(\\theta_{\\text{merged}}, T_i)}{P(\\theta_i, T_i)} - 1\\right) \\times 100\\%

    where :math:`P(\\theta_i, T_i)` is the performance of the individual fine-tuned
    model on task :math:`T_i`, and :math:`P(\\theta_{\\text{merged}}, T_i)` is the
    performance of the merged model on the same task. The result lies in
    :math:`[-100\\%, 0\\%]`, where 0% means no degradation.

    Args:
        merged_performance: Performance of the merged model. Either a single scalar
            for one task, or a dict mapping task names to performance values.
        individual_performance: Performance of the individual fine-tuned model(s).
            Either a single scalar for one task, or a dict mapping task names to
            performance values. Must match the type/keys of ``merged_performance``.
        task_names: Optional list of task names to restrict computation to a subset
            of keys when both inputs are dicts.
        reduction: Optional reduction method to apply when inputs are dicts. If
            ``"mean"``, returns the average merging loss across tasks instead of a dict.

    Returns:
        The merging loss as a percentage in :math:`[-100, 0]`. Returns a single
        float when scalar inputs are given, or a dict of floats keyed by task name
        when dict inputs are given.

    Raises:
        ValueError: If ``individual_performance`` is zero for any task (division by
            zero), or if dict inputs have mismatched keys.

    Examples:
        Single task::

            >>> compute_merging_loss(merged_performance=0.85, individual_performance=0.95)
            -10.526315789473685

        Multiple tasks::

            >>> compute_merging_loss(
            ...     merged_performance={"cars": 0.80, "dtd": 0.70},
            ...     individual_performance={"cars": 0.90, "dtd": 0.75},
            ... )
            {'cars': -11.11111111111111, 'dtd': -6.666666666666667}
    """
    if isinstance(merged_performance, dict) and isinstance(
        individual_performance, dict
    ):
        keys = task_names if task_names is not None else list(merged_performance.keys())
        missing = set(keys) - merged_performance.keys()
        if missing:
            raise ValueError(f"Keys missing from merged_performance: {missing}")
        missing = set(keys) - individual_performance.keys()
        if missing:
            raise ValueError(f"Keys missing from individual_performance: {missing}")
        result = {
            task: _merging_loss_scalar(
                merged_performance[task], individual_performance[task]
            )
            for task in keys
        }
        if reduction is not None:
            if reduction == "mean":
                return sum(result.values()) / len(result)
            else:
                raise ValueError(f"Unsupported reduction method: {reduction}")
        return result
    else:
        return _merging_loss_scalar(
            float(merged_performance), float(individual_performance)
        )
