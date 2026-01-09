"""
functions deal with tensorboard logs.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def parse_tensorboard_as_dict(
    path: Union[str, Path],
    scalars: Iterable[str],
) -> Dict[str, pd.DataFrame]:
    """
    returns a dictionary of pandas dataframes for each requested scalar.

    Args:
        path(str): A file path to a directory containing tf events files, or a single
                   tf events file. The accumulator will load events from this path.
        scalars:   scalars

    Returns:
        Dict[str, pandas.DataFrame]: a dictionary of pandas dataframes for each requested scalar

    Example:

        >>> from fusion_bench.utils.tensorboard import parse_tensorboard_as_dict
        >>> path = "path/to/tensorboard/logs"
        >>> scalars = ["train/loss", "val/accuracy"]
        >>> data = parse_tensorboard_as_dict(path, scalars)
        >>> train_loss_df = data["train/loss"]
        >>> val_accuracy_df = data["val/accuracy"]
    """
    if isinstance(path, Path):
        path = str(path)
    assert isinstance(path, str), "path must be a string"
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


def parse_tensorboard_as_list(
    path: Union[str, Path], scalars: Iterable[str]
) -> List[pd.DataFrame]:
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
