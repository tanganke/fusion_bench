"""
functions deal with tensorboard logs.
"""

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def parse_tensorboard_as_dict(path: str, scalars: Iterable[str]):
    """
    returns a dictionary of pandas dataframes for each requested scalar.

    Args:
        path(str): A file path to a directory containing tf events files, or a single
                   tf events file. The accumulator will load events from this path.
        scalars:   scalars

    Returns:
        Dict[str, pandas.DataFrame]: a dictionary of pandas dataframes for each requested scalar
    """
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}


def parse_tensorboard_as_list(path: str, scalars: Iterable[str]):
    """
    returns a list of pandas dataframes for each requested scalar.

    see also: :py:func:`parse_tensorboard_as_dict`

    Args:
        path(str): A file path to a directory containing tf events files, or a single
                   tf events file. The accumulator will load events from this path.
        scalars:   scalars

    Returns:
        List[pandas.DataFrame]: a list of pandas dataframes for each requested scalar.
    """
    d = parse_tensorboard_as_dict(path, scalars)
    return [d[s] for s in scalars]
