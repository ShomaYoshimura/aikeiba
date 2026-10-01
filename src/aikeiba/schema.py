"""Column names for the runner table: one row per horse per race.

Every module takes and returns DataFrames in this shape so that the
pipeline stages can be composed without adapters.
"""

from __future__ import annotations

import pandas as pd

RACE_ID = "race_id"
RACE_DATE = "race_date"
POST_TIME = "post_time"  # scheduled race start (tz-naive, JST)
HORSE_ID = "horse_id"
FINISH_POSITION = "finish_position"  # 1 = winner
WIN_ODDS = "win_odds"  # decimal win odds available at prediction time
GRADE = "grade"  # "G1", "G2", "G3", or anything else for non-graded races

REQUIRED_COLUMNS = (RACE_ID, RACE_DATE, POST_TIME, HORSE_ID, FINISH_POSITION, WIN_ODDS, GRADE)
GRADED = ("G1", "G2", "G3")


def validate_runners(df: pd.DataFrame) -> None:
    """Raise ValueError if ``df`` does not satisfy the runner-table contract."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if df.duplicated([RACE_ID, HORSE_ID]).any():
        raise ValueError("duplicate (race_id, horse_id) rows")
    if (df[FINISH_POSITION] < 1).any():
        raise ValueError("finish_position must be >= 1")
    if (df[WIN_ODDS] < 1.0).any():
        raise ValueError("win_odds must be decimal odds >= 1.0")
    winners = df[df[FINISH_POSITION] == 1].groupby(RACE_ID).size()
    no_winner = set(df[RACE_ID].unique()) - set(winners.index)
    if no_winner:
        raise ValueError(f"{len(no_winner)} races have no winner, e.g. {sorted(no_winner)[:3]}")


def is_graded(df: pd.DataFrame) -> pd.Series:
    return df[GRADE].isin(GRADED)
