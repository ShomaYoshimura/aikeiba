"""Point-in-time guarantees: a feature may only use data available before post time."""

from __future__ import annotations

import pandas as pd

from aikeiba.schema import POST_TIME, RACE_ID


class LeakageError(ValueError):
    """Raised when a value would not have been available at prediction time."""


def assert_point_in_time(
    df: pd.DataFrame,
    available_at_col: str,
    post_time_col: str = POST_TIME,
    *,
    allow_missing: bool = True,
) -> None:
    """Fail if any value in ``available_at_col`` is at or after the race's post time.

    ``allow_missing`` treats NaT as "no information" (e.g. a debut runner has no
    prior form), which is safe. Set it to False for values that must always exist.
    """
    available_at = df[available_at_col]
    missing = available_at.isna()
    if missing.any() and not allow_missing:
        raise LeakageError(f"{int(missing.sum())} rows have no {available_at_col}")
    late = ~missing & (available_at >= df[post_time_col])
    if late.any():
        races = df.loc[late, RACE_ID].unique()[:5].tolist()
        raise LeakageError(
            f"{int(late.sum())} rows use {available_at_col} at or after post time, "
            f"e.g. races {races}"
        )
