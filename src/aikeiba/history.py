"""History table: past races, one row per runner, used to abstract past results.

It extends the runner table in ``schema.py`` with the connections (jockey, trainer,
owner, pedigree, breeder) and race conditions needed to build the factors. Only
races strictly before the target race's date are ever used for a prediction.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from aikeiba.conditions import GOING_GROUP, condition_bin, race_name_key
from aikeiba.history_metrics import race_levels, speed_figures
from aikeiba.racecard import (
    COURSE_TREND_ATTRS,
    EDITION_TREND_ATTRS,
    GOING_ALIASES,
    SURFACE_ALIASES,
    PastRun,
    RaceCard,
    Record,
    Runner,
    last_finish_band,
)
from aikeiba.schema import FINISH_POSITION, GRADED, HORSE_ID, POST_TIME, RACE_DATE, RACE_ID
from aikeiba.stats import head_to_head

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
    "time",  # final time in seconds
    "early_position",  # position at the first corner
    "cushion",
    "moisture",
    "trouble",  # 1 if the run had a bad start or interference
    "dam",
    "trainer_center",  # 美浦 / 栗東 (or east / west)
    "prev_odds",  # odds observed earlier than win_odds (e.g. the day before)
    "popularity",  # betting rank; derived from win_odds when missing
    "weather",
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
    if h["popularity"].isna().all() and h["win_odds"].notna().any():
        h["popularity"] = h.groupby(RACE_ID)["win_odds"].rank(method="min")
    h["margin"] = h["time"] - h.groupby(RACE_ID)["time"].transform("min")
    h["post_time"] = pd.to_datetime(h["post_time"])
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
            dam=opt(r, "dam"),
            damsire=opt(r, "damsire"),
            trainer_center=opt(r, "trainer_center"),
            previous_odds=float(r["prev_odds"]) if has_odds and pd.notna(r["prev_odds"]) else None,
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
        post_time=(
            None if pd.isna(first[POST_TIME]) else pd.Timestamp(first[POST_TIME]).strftime("%H:%M")
        ),
        cushion=opt(first, "cushion"),
        moisture=opt(first, "moisture"),
        weather=opt(first, "weather"),
        runners=runners,
    )


# --- history -> race card -------------------------------------------------------------
# Training and race day share one path: history fills the card's own fields (the ones the
# collector agents fill from public sources), and factors are computed from the card only.

EDITION_YEARS = 10  # past editions of the race used for ``trends``
REFERENCE_YEARS = 5  # reference races at the same course and distance for ``course_trends``
_RUN_FIELDS = (
    "date", "finish", "field_size", "race_name", "course", "surface", "distance", "going",
    "grade", "jockey", "last3f", "time", "margin", "early_position", "speed_figure",
    "race_level", "cushion", "trouble", "popularity", "weight_carried",
)  # fmt: skip
_HISTORY_COLUMN = {"date": RACE_DATE, "finish": FINISH_POSITION}
_INT_FIELDS = {"finish", "field_size", "distance", "early_position", "popularity"}


def _plain(value, as_int: bool = False):
    """A history cell as a plain Python value for a card field (NaN -> None)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        value = value.item()
    return int(value) if as_int else value


def _past_runs(rows: pd.DataFrame) -> tuple[PastRun, ...]:
    out = []
    for _, r in rows.sort_values(RACE_DATE, ascending=False, kind="stable").iterrows():
        run = {f: _plain(r[_HISTORY_COLUMN.get(f, f)], f in _INT_FIELDS) for f in _RUN_FIELDS}
        if run["trouble"] is not None:
            run["trouble"] = bool(run["trouble"])
        out.append(PastRun(**run))
    return tuple(out)


def _counts(rows: pd.DataFrame, keys) -> dict:
    """Group key (a value, or a tuple for several keys) -> Record(starts, top3)."""
    if rows.empty:
        return {}
    g = rows.groupby(keys, observed=True)["top3"].agg(["size", "sum"])
    return {k: Record(int(n), int(t)) for k, (n, t) in zip(g.index, g.to_numpy(), strict=True)}


def _trend_key(value) -> str | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return str(int(value)) if isinstance(value, float | np.floating) else str(value)


def _trend_records(rows: pd.DataFrame, attributes) -> dict:
    """attribute -> value -> Record, the shape of a card's ``trends`` / ``course_trends``."""
    out = {}
    for attr in attributes:
        if attr == "last_finish":
            keys = rows["prev_finish"].map(last_finish_band)
        else:
            keys = rows[attr].map(_trend_key)
        g = pd.DataFrame({"key": keys, "top3": rows["top3"]}).dropna()
        if not g.empty:
            out[attr] = _counts(g, "key")
    return out


def _same_day_races(card: RaceCard, before_today: pd.DataFrame) -> tuple:
    """Earlier races today on this course and surface (needs the card's post time)."""
    if card.post_time is None:
        return ()
    start = pd.Timestamp(f"{card.date} {card.post_time}")
    today = before_today[
        (before_today["course"] == card.course)
        & (before_today["surface"] == card.surface)
        & (before_today[POST_TIME] < start)
    ]
    return tuple(
        {
            "surface": card.surface,
            "distance": _plain(rows["distance"].iloc[0], True),
            "runners": [
                {
                    "draw": _plain(r.draw, True),
                    "early_position": _plain(r.early_position, True),
                    "finish": int(r.finish_position),
                }
                for r in rows.itertuples()
            ],
        }
        for _, rows in today.groupby(RACE_ID, sort=False)
    )


def enrich_card(card: RaceCard, history: pd.DataFrame) -> RaceCard:
    """The card with its fields filled from history, as known before the race.

    Fills each runner's ``past_runs`` (with speed figures and race levels computed as of
    the race date) and ``stats``, and the card's ``trends``, ``course_trends``,
    ``course_draw_stats`` and ``same_day_races``: the same fields, in the same form, that
    the collector agents fill from public sources. Only rows dated before the race day are
    used (plus earlier races on the race day itself for ``same_day_races``). Where history
    has nothing, the card's own value is kept. ``factors.compute_factors`` then reads the
    card only, so training (``train.build_examples``) and race day use one path.
    """
    if "cond_bin" not in history.columns:
        history = prepare_history(history)

    today = pd.Timestamp(card.date)
    # history is sorted by date, so "before the race day" is a prefix
    cut = history[RACE_DATE].searchsorted(today, side="left")
    before = history.iloc[:cut]
    same_day = card.same_day_races
    if not same_day:
        end = history[RACE_DATE].searchsorted(today, side="right")
        same_day = _same_day_races(card, history.iloc[cut:end])
    if before.empty:
        return replace(card, same_day_races=same_day)

    r = card.runners
    ids = [x.horse_id for x in r if x.horse_id is not None]
    horse_rows = before[before[HORSE_ID].isin(ids) | before["horse"].isin([x.horse for x in r])]
    horse_rows = horse_rows.assign(
        speed_figure=speed_figures(horse_rows, before), race_level=race_levels(horse_rows, before)
    )

    recent3 = before[before[RACE_DATE] >= today - pd.DateOffset(years=3)]
    recent5 = before[before[RACE_DATE] >= today - pd.DateOffset(years=5)]
    graded3 = recent3[recent3["grade"].isin(GRADED)]
    course = before[(before["course"] == card.course) & (before["surface"] == card.surface)]
    same_track = course[course["distance"] == card.distance]
    cond = condition_bin(card.surface, card.distance, card.going)

    def entity(rows, col):
        values = {getattr(x, col) for x in r} - {None}
        return _counts(rows[rows[col].isin(values)], col)

    tables = {
        "jockey_year": ("jockey", entity(before[before["year"] == today.year], "jockey")),
        "jockey_graded": ("jockey", entity(graded3, "jockey")),
        "jockey_course": ("jockey", entity(course, "jockey")),
        "trainer": ("trainer", entity(recent3, "trainer")),
        "trainer_graded": ("trainer", entity(graded3, "trainer")),
        "owner": ("owner", entity(recent3, "owner")),
        "breeder": ("breeder", entity(recent5, "breeder")),
        "region": ("region", entity(recent5, "region")),
        "sire_condition": ("sire", entity(before[before["cond_bin"] == cond], "sire")),
        "damsire_condition": ("damsire", entity(before[before["cond_bin"] == cond], "damsire")),
    }
    trainers = {x.trainer for x in r} - {None}
    jockeys = [x.jockey for x in r if x.jockey is not None]
    pairs = _counts(
        before[before["trainer"].isin(trainers) & before["jockey"].isin(jockeys)],
        ["trainer", "jockey"],
    )
    h2h = head_to_head(recent3, "jockey", jockeys) if jockeys else {}
    dams = {x.dam for x in r} - {None}
    foals = before[before["dam"].isin(dams)]
    dams_raced = _counts(before[before["horse"].isin(dams)], "horse")

    runners = []
    for x in r:
        if x.horse_id is not None:
            own_rows = horse_rows[horse_rows[HORSE_ID] == x.horse_id]
        else:
            own_rows = horse_rows[horse_rows["horse"] == x.horse]
        stats = {}
        for key, (attr, table) in tables.items():
            value = getattr(x, attr)
            if value is not None and value in table:
                stats[key] = table[value]
        if (x.trainer, x.jockey) in pairs:
            stats["trainer_jockey"] = pairs[(x.trainer, x.jockey)]
        if x.jockey is not None and not pd.isna(h2h.get(x.jockey, np.nan)):
            stats["jockey_h2h"] = float(h2h[x.jockey])
        if x.dam is not None:
            own = (foals[HORSE_ID] == x.horse_id) | (foals["horse"] == x.horse)
            siblings = foals[(foals["dam"] == x.dam) & ~own]
            if not siblings.empty:
                stats["siblings"] = Record(len(siblings), int(siblings["top3"].sum()))
            if x.dam in dams_raced:
                stats["dam_race"] = dams_raced[x.dam]
        runners.append(
            replace(
                x,
                past_runs=_past_runs(own_rows) if not own_rows.empty else x.past_runs,
                stats={**x.stats, **stats},
            )
        )

    editions = before[before["name_key"] == race_name_key(card.name)]
    editions = editions[editions[RACE_DATE] >= today - pd.DateOffset(years=EDITION_YEARS)]
    reference = same_track[same_track[RACE_DATE] >= today - pd.DateOffset(years=REFERENCE_YEARS)]
    gates = _counts(same_track.dropna(subset=["draw"]), "draw")
    draws = {str(int(k)): rec for k, rec in gates.items()}
    return replace(
        card,
        runners=tuple(runners),
        trends=_trend_records(editions, EDITION_TREND_ATTRS) or card.trends,
        course_trends=_trend_records(reference, COURSE_TREND_ATTRS) or card.course_trends,
        course_draw_stats=draws or card.course_draw_stats,
        same_day_races=same_day,
    )
