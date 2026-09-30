"""Per-horse form features computed strictly from races before the current one."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aikeiba.leakage import assert_point_in_time
from aikeiba.schema import FINISH_POSITION, HORSE_ID, POST_TIME, RACE_ID

FORM_FEATURES = (
    "prior_starts",
    "prior_win_rate",
    "prior_top3_rate",
    "prior_mean_finish_pct",
    "last_finish_pct",
    "days_since_last",
)
FORM_AVAILABLE_AT = "form_available_at"


def add_prior_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``FORM_FEATURES`` and ``form_available_at`` added.

    Each feature for a runner aggregates only that horse's earlier races, so the
    information was known once the previous race finished. ``form_available_at`` is
    the previous race's post time (NaT for a debut) and is checked against the
    current post time before returning.
    """
    out = df.copy()
    field_size = out.groupby(RACE_ID)[HORSE_ID].transform("size")
    # 0 = winner, 1 = last place; comparable across field sizes.
    finish_pct = (out[FINISH_POSITION] - 1) / (field_size - 1).clip(lower=1)
    win = (out[FINISH_POSITION] == 1).astype(float)
    top3 = (out[FINISH_POSITION] <= 3).astype(float)

    order = out.sort_values([HORSE_ID, POST_TIME]).index
    g = pd.DataFrame(
        {
            HORSE_ID: out.loc[order, HORSE_ID],
            "post_time": out.loc[order, POST_TIME],
            "finish_pct": finish_pct.loc[order],
            "win": win.loc[order],
            "top3": top3.loc[order],
        }
    )
    by_horse = g.groupby(HORSE_ID, sort=False)
    starts = by_horse.cumcount()
    # cumsum minus the current row = sum over strictly earlier races
    prior = {c: by_horse[c].cumsum() - g[c] for c in ("finish_pct", "win", "top3")}
    denom = starts.replace(0, np.nan)
    prev_post = by_horse["post_time"].shift(1)

    feats = pd.DataFrame(
        {
            "prior_starts": starts.astype(float),
            "prior_win_rate": prior["win"] / denom,
            "prior_top3_rate": prior["top3"] / denom,
            "prior_mean_finish_pct": prior["finish_pct"] / denom,
            "last_finish_pct": by_horse["finish_pct"].shift(1),
            "days_since_last": (g["post_time"] - prev_post).dt.total_seconds() / 86400,
            FORM_AVAILABLE_AT: prev_post,
        },
        index=order,
    )
    out = out.join(feats)
    assert_point_in_time(out, FORM_AVAILABLE_AT)
    return out
