"""Per-run metrics derived from the history table as of a cutoff.

Both functions take ``rows`` (past runs, a subset of ``before``) and ``before`` (all
history before the cutoff), and use nothing after the cutoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aikeiba.stats import shrunk_rate

_TRACK = ["course", "surface", "distance"]
_DAY = ["race_date", "course", "surface"]
_KG_POINTS = 1.0  # one figure point per kg carried above 55 kg (about 0.1 s per 1000 m)


def speed_figures(rows: pd.DataFrame, before: pd.DataFrame) -> pd.Series:
    """Speed figure per run: 100 + 10 x (seconds per 1000 m faster than the standard,
    after removing the day's track variant) + weight carried adjustment.

    The standard for a course, surface and distance is the median time of the first three
    finishers; the track variant of a day is the median deviation of that day's first three
    finishers from their standards.
    """
    out = pd.Series(np.nan, index=rows.index)
    if rows.empty or rows["time"].isna().all():
        return out
    timed = before[before["time"].notna() & (before["finish_position"] <= 3)]
    day_rows = timed.merge(rows[_DAY].drop_duplicates(), on=_DAY)
    keys = pd.concat([rows[_TRACK], day_rows[_TRACK]]).drop_duplicates()
    std = timed.merge(keys, on=_TRACK).groupby(_TRACK)["time"].median().rename("standard")
    day_rows = day_rows.join(std, on=_TRACK)
    day_rows["dev"] = (day_rows["standard"] - day_rows["time"]) * 1000 / day_rows["distance"]
    variant = day_rows.groupby(_DAY)["dev"].median().rename("variant")
    r = rows.join(std, on=_TRACK).join(variant, on=_DAY)
    dev = (r["standard"] - r["time"]) * 1000 / r["distance"] - r["variant"].fillna(0.0)
    weight = pd.to_numeric(r["weight_carried"], errors="coerce").fillna(55.0)
    return 100 + 10 * dev + (weight - 55.0) * _KG_POINTS


def race_levels(rows: pd.DataFrame, before: pd.DataFrame, strength: float = 5.0) -> pd.Series:
    """Opponent quality per run: the shrunk top-3 rate of the *other* runners of that race
    in their later races (still before the cutoff)."""
    out = pd.Series(np.nan, index=rows.index)
    if rows.empty:
        return out
    members = before.loc[
        before["race_id"].isin(rows["race_id"].unique()), ["race_id", "horse_id", "race_date"]
    ]
    later = before.loc[
        before["horse_id"].isin(members["horse_id"].unique()), ["horse_id", "race_date", "top3"]
    ]
    m = members.merge(later, on="horse_id", suffixes=("", "_later"))
    m = m[m["race_date_later"] > m["race_date"]]
    if m.empty:
        return out
    per_horse = m.groupby(["race_id", "horse_id"])["top3"].agg(["sum", "size"])
    per_race = per_horse.groupby(level="race_id").sum()
    own = per_horse.reindex(pd.MultiIndex.from_frame(rows[["race_id", "horse_id"]])).fillna(0)
    total = per_race.reindex(rows["race_id"]).fillna(0)
    successes = total["sum"].to_numpy() - own["sum"].to_numpy()
    trials = total["size"].to_numpy() - own["size"].to_numpy()
    rates = shrunk_rate(successes, trials, float(before["top3"].mean()), strength)
    return pd.Series(np.where(trials > 0, rates, np.nan), index=rows.index)
