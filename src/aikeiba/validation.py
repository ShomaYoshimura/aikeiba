"""Walk-forward (rolling-origin) splits by calendar year."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    test_year: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def walk_forward_by_year(
    race_dates,
    test_years: Iterable[int],
    *,
    max_train_years: int | None = None,
) -> list[Fold]:
    """For each test year, train on every earlier year (or the last ``max_train_years``).

    Indices are positions into ``race_dates``. Years with no training data or no
    test data are skipped, so callers can pass a generous range.
    """
    years = pd.DatetimeIndex(pd.to_datetime(np.asarray(race_dates))).year.to_numpy()
    folds = []
    for year in test_years:
        train = years < year
        if max_train_years is not None:
            train &= years >= year - max_train_years
        test = years == year
        if train.any() and test.any():
            folds.append(Fold(year, np.flatnonzero(train), np.flatnonzero(test)))
    return folds
