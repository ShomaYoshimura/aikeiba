"""Synthetic runner table so the pipeline runs end to end without licensed data.

Horses have a latent ability that drifts slowly. The market sees a noisy estimate
of current ability; the model only sees past results. No model edge is built in
deliberately, so the backtest is a smoke test of the pipeline, not evidence of skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aikeiba.schema import (
    FINISH_POSITION,
    GRADE,
    HORSE_ID,
    POST_TIME,
    RACE_DATE,
    RACE_ID,
    WIN_ODDS,
)

TAKEOUT = 0.20  # JRA win-pool takeout


def generate_runners(
    n_days: int = 400,
    races_per_day: int = 12,
    n_horses: int = 2500,
    field_size: tuple[int, int] = (8, 18),
    start: str = "2019-01-05",
    market_noise: float = 0.35,
    race_noise: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ability = rng.normal(0.0, 1.0, n_horses)
    days = pd.Timestamp(start) + pd.to_timedelta(np.arange(n_days) * 3, unit="D")

    rows = []
    for day in days:
        ability += rng.normal(0.0, 0.03, n_horses)  # slow drift in form
        sizes = rng.integers(field_size[0], field_size[1] + 1, races_per_day)
        horses = rng.choice(n_horses, size=sizes.sum(), replace=False)
        for r, runners in enumerate(np.split(horses, np.cumsum(sizes)[:-1])):
            perf = ability[runners] + rng.normal(0.0, race_noise, len(runners))
            finish = np.empty(len(runners), dtype=int)
            finish[np.argsort(-perf)] = np.arange(1, len(runners) + 1)
            est = ability[runners] + rng.normal(0.0, market_noise, len(runners))
            p_mkt = np.exp(1.6 * est)
            p_mkt /= p_mkt.sum()
            odds = np.maximum(1.0, np.round((1.0 - TAKEOUT) / p_mkt, 1))
            race_id = f"{day:%Y%m%d}{r + 1:02d}"
            post = day + pd.Timedelta(hours=10, minutes=r * 30)
            for h, pos, o in zip(runners, finish, odds, strict=True):
                rows.append((race_id, day, post, f"H{h:05d}", pos, o, ability[runners].mean()))

    df = pd.DataFrame(
        rows,
        columns=[RACE_ID, RACE_DATE, POST_TIME, HORSE_ID, FINISH_POSITION, WIN_ODDS, "_field"],
    )
    # Strongest fields become graded races, roughly the JRA share (~3% of races).
    field = df.groupby(RACE_ID)["_field"].first()
    grade = pd.cut(
        field.rank(pct=True),
        bins=[0, 0.97, 0.98, 0.99, 1.0],
        labels=["OP", "G3", "G2", "G1"],
    ).astype(str)
    df[GRADE] = df[RACE_ID].map(grade)
    return df.drop(columns="_field")
