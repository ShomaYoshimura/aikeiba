"""History table: past races, one row per runner, used to abstract past results.

It extends the runner table in ``schema.py`` with the connections (jockey, trainer,
owner, pedigree, breeder) and race conditions needed to build the factors. Only
races strictly before the target race's date are ever used for a prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aikeiba.conditions import GOING_GROUP, condition_bin, race_name_key
from aikeiba.racecard import GOING_ALIASES, SURFACE_ALIASES, RaceCard, Runner
from aikeiba.schema import FINISH_POSITION, HORSE_ID, POST_TIME, RACE_DATE, RACE_ID

REQUIRED = (
    RACE_ID,
    RACE_DATE,
    "race_name",
    "course",
    "surface",
    "distance",
    "going",
    HORSE_ID,
    "horse",
    FINISH_POSITION,
)
OPTIONAL = (
    POST_TIME,
    "grade",
    "number",
    "draw",
    "win_odds",
    "jockey",
    "trainer",
    "owner",
    "sire",
    "damsire",
    "breeder",
    "region",
    "age",
    "sex",
    "weight_carried",
    "horse_weight",
    "horse_weight_change",
    "running_style",
    "last3f",
)


def prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    """Validate, normalize and add the derived columns the factors need.

    Adds field_size, top3, cond_bin, going_group, name_key, year and prev_finish (the
    horse's finish in its previous race), and sorts by date.
    """
    missing = [c for c in REQUIRED if c not in history.columns]
    if missing:
        raise ValueError(f"history is missing columns: {missing}")
    h = history.copy()
    for col in OPTIONAL:
        if col not in h.columns:
            h[col] = np.nan
    h[RACE_DATE] = pd.to_datetime(h[RACE_DATE])
    h["going"] = h["going"].map(lambda g: GOING_ALIASES.get(g, g))
    h["surface"] = h["surface"].map(lambda s: SURFACE_ALIASES.get(s, s))
    unknown = set(h["going"].unique()) - set(GOING_GROUP)
    if unknown:
        raise ValueError(f"unknown going values in history: {sorted(unknown)}")
    h["field_size"] = h.groupby(RACE_ID)[HORSE_ID].transform("size")
    h["top3"] = (h[FINISH_POSITION] <= 3).astype(int)
    h["going_group"] = h["going"].map(GOING_GROUP)
    bins = {
        key: condition_bin(*key)
        for key in h[["surface", "distance", "going"]].drop_duplicates().itertuples(index=False)
    }
    h["cond_bin"] = [bins[k] for k in h[["surface", "distance", "going"]].itertuples(index=False)]
    h["name_key"] = h["race_name"].map(race_name_key)
    h["year"] = h[RACE_DATE].dt.year
    h = h.sort_values([RACE_DATE, RACE_ID], kind="stable").reset_index(drop=True)
    h["prev_finish"] = h.groupby(HORSE_ID)[FINISH_POSITION].shift(1)
    return h


def card_from_history(rows: pd.DataFrame) -> RaceCard:
    """Race card for a past race, with only what was known before it (no results, no stats).

    Odds are included when every runner has them. Factors are then computed from the
    history table, restricted to earlier dates.
    """
    first = rows.iloc[0]
    has_odds = rows["win_odds"].notna().all()

    def opt(r, col):
        v = r[col]
        return None if pd.isna(v) else v

    runners = tuple(
        Runner(
            number=int(r["number"]) if pd.notna(r["number"]) else i + 1,
            horse=r["horse"],
            horse_id=r[HORSE_ID],
            draw=opt(r, "draw"),
            jockey=opt(r, "jockey"),
            trainer=opt(r, "trainer"),
            owner=opt(r, "owner"),
            sire=opt(r, "sire"),
            damsire=opt(r, "damsire"),
            breeder=opt(r, "breeder"),
            region=opt(r, "region"),
            age=opt(r, "age"),
            sex=opt(r, "sex"),
            weight_carried=opt(r, "weight_carried"),
            horse_weight=opt(r, "horse_weight"),
            horse_weight_change=opt(r, "horse_weight_change"),
            win_odds=float(r["win_odds"]) if has_odds else None,
            running_style=opt(r, "running_style"),
        )
        for i, (_, r) in enumerate(rows.iterrows())
    )
    return RaceCard(
        name=first["race_name"],
        date=pd.Timestamp(first[RACE_DATE]).strftime("%Y-%m-%d"),
        course=first["course"],
        surface=first["surface"],
        distance=int(first["distance"]),
        going=first["going"],
        grade=opt(first, "grade"),
        runners=runners,
    )
