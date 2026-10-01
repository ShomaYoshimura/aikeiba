"""Turn everything known about a race into one numeric vector per runner.

``compute_factors`` returns raw values (NaN = unknown). ``model.standardize`` then
z-scores each factor within the race, so every factor is expressed as "how much
better than this field" and missing values count as the field average.

With a history table the statistics are computed from it, using only races on
earlier dates. Without one, they come from the card's ``past_runs``, ``stats`` and
``trends`` fields, which the predict-race skill fills from public sources.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aikeiba.bias import estimate_bias, inside_score
from aikeiba.conditions import GOING_GROUP, condition_bin, course_region, race_name_key, similarity
from aikeiba.history_metrics import race_levels, speed_figures
from aikeiba.racecard import (
    COURSE_TREND_ATTRS,
    EAST_WEST,
    EDITION_TREND_ATTRS,
    WORKOUT_RATINGS,
    RaceCard,
    Runner,
    inbreeding_coefficient,
)
from aikeiba.schema import GRADED
from aikeiba.stats import (
    grouped_top3_rates,
    head_to_head,
    profile_matrix,
    shrunk_rate,
    top3_rate,
)


@dataclass(frozen=True)
class Factor:
    name: str
    group: str
    description: str
    prior_weight: float  # log-strength per within-race standard deviation, before fitting


FACTORS = (
    # Horse: past performance
    Factor("form", "horse", "decay-weighted performance over the last 5 runs", 0.25),
    Factor("class_form", "horse", "performance weighted by race class (G1 > G2 > G3)", 0.25),
    Factor("similar_form", "horse", "performance in runs similar to today's race", 0.15),
    Factor("going_aptitude", "horse", "top-3 rate on today's going group", 0.08),
    Factor("course_aptitude", "horse", "top-3 rate at this course and distance (±200 m)", 0.08),
    Factor("closing_speed", "horse", "negated mean final-600 m time, same surface", 0.08),
    Factor("distance_change", "horse", "absolute log change from the last run's distance", -0.03),
    Factor("layoff", "horse", "1 if the last run was 90+ days ago", -0.05),
    Factor("weight_carried", "horse", "weight carried (kg)", 0.0),
    Factor("horse_weight_change", "horse", "absolute body-weight change (kg)", -0.03),
    Factor("age", "horse", "age in years", -0.03),
    Factor("jockey_change", "horse", "1 if the jockey differs from the last run", 0.0),
    Factor("speed_index", "horse", "decay-weighted speed figure over the last 5 runs", 0.20),
    Factor("margin_form", "horse", "negated decay-weighted seconds behind the winner", 0.12),
    Factor("early_speed", "horse", "how far forward it races early (-1 last .. 1 leading)", 0.03),
    Factor("race_level", "horse", "opponents' later top-3 rate in its recent races", 0.08),
    Factor("cushion_fit", "horse", "performance on cushion values close to today's", 0.03),
    Factor("recent_trouble", "horse", "share of the last 5 runs with trouble (an excuse)", 0.03),
    Factor("market_gap", "horse", "how much better it finished than its betting rank", 0.02),
    Factor("runs_since_layoff", "horse", "runs since the last 90+ day break (0 = first back)", 0.0),
    Factor("last_race_class", "horse", "class of the last race (G1 > G2 > G3 > other)", 0.04),
    Factor("days_since_last", "horse", "log(1 + days since the last run)", 0.0),
    Factor("first_course", "horse", "1 if it has never run at this course and surface", -0.02),
    Factor("first_distance", "horse", "1 if it has never run within 100 m on this surface", -0.02),
    Factor("weight_ratio", "horse", "weight carried / body weight", -0.02),
    Factor("summer_mare", "horse", "1 for a filly or mare in June-September", 0.0),
    Factor("autumn_three_year_old", "horse", "1 for a 3-year-old from September", 0.0),
    # Pedigree and connections
    Factor("sire_aptitude", "pedigree", "sire's progeny top-3 rate in today's condition", 0.06),
    Factor("damsire_aptitude", "pedigree", "damsire's progeny rate in today's condition", 0.04),
    Factor("sibling_rate", "pedigree", "top-3 rate of the dam's other foals", 0.04),
    Factor("dam_race_rate", "pedigree", "the dam's own top-3 rate as a racehorse", 0.01),
    Factor("inbreeding", "pedigree", "inbreeding coefficient from the listed crosses", 0.0),
    Factor("breeder_rate", "connections", "breeder's top-3 rate (5 years)", 0.03),
    Factor("region_rate", "connections", "top-3 rate of horses from the same region", 0.02),
    Factor("trainer_rate", "connections", "trainer's top-3 rate (3 years)", 0.08),
    Factor("trainer_graded_rate", "connections", "trainer's top-3 rate in graded races", 0.06),
    Factor("trainer_jockey_rate", "connections", "top-3 rate of this trainer-jockey pair", 0.04),
    Factor("owner_rate", "connections", "owner's top-3 rate (3 years)", 0.04),
    Factor("long_shipping", "connections", "1 if the trainer's centre is on the other side", -0.02),
    # Jockey
    Factor("jockey_year_rate", "jockey", "jockey's top-3 rate this year", 0.10),
    Factor("jockey_graded_rate", "jockey", "jockey's top-3 rate in graded races (3 years)", 0.08),
    Factor("jockey_course_rate", "jockey", "jockey's top-3 rate at this course", 0.04),
    Factor("jockey_h2h", "jockey", "share of races finished ahead of the field's jockeys", 0.06),
    Factor("jockey_horse_rate", "jockey", "top-3 rate of this jockey on this horse", 0.04),
    Factor("first_ride", "jockey", "1 if this jockey has never ridden the horse", -0.02),
    # Race
    Factor("draw_bias", "race", "top-3 rate of this gate at this course and distance", 0.05),
    Factor("race_trend", "race", "fit to past editions (gate, age, sex, style, last run)", 0.06),
    Factor(
        "course_trend",
        "race",
        "fit to races at this course, surface and distance (style, age, sex, last run, sire)",
        0.05,
    ),
    Factor("track_bias_today", "race", "fit to today's inside/outside and pace bias", 0.05),
    # Current information
    Factor("workout_score", "current", "latest workout vs. the field (time or A-E rating)", 0.06),
    Factor("comment_score", "current", "stable comment, scored -2..2", 0.05),
    Factor("consensus_share", "current", "share of public predictions marking the horse", 0.03),
    Factor("paddock_score", "current", "paddock and warm-up impression, scored -2..2", 0.03),
    Factor(
        "odds_drift", "current", "log(earlier odds / current odds): shortening is positive", 0.03
    ),
)
FACTOR_NAMES = tuple(f.name for f in FACTORS)

_BASE_TOP3 = 0.2  # prior top-3 rate for an average runner (3 of ~15)
_DECAY = 0.8
_CLASS_WEIGHT = {"G1": 1.0, "G2": 0.85, "G3": 0.7}
_OTHER_CLASS_WEIGHT = 0.5
# pseudo-counts for shrinkage
_HORSE_M, _ENTITY_M, _PAIR_M, _PROFILE_M, _DRAW_M, _TREND_M = 3, 20, 10, 30, 50, 10
# Attributes compared against past editions of the race and against reference races at the
# same course and distance. The gate is left out of the latter: draw_bias already covers it.
RACE_TREND_ATTRS = EDITION_TREND_ATTRS
_EDITION_YEARS, _REFERENCE_YEARS = 10, 5


RUN_COLUMNS = (
    "date", "finish", "field_size", "course", "surface", "distance", "going", "grade",
    "jockey", "last3f", "time", "margin", "early_position", "speed_figure", "race_level",
    "cushion", "trouble", "popularity", "weight_carried",
)  # fmt: skip
# history column for each run column, where the names differ
_HISTORY_RUN_COLUMNS = {"date": "race_date", "finish": "finish_position"}
FEMALE = {"F", "M", "牝", "filly", "mare"}


def _past_runs_frame(runner: Runner) -> pd.DataFrame:
    if not runner.past_runs:
        return pd.DataFrame(columns=list(RUN_COLUMNS))
    df = pd.DataFrame([vars(p) for p in runner.past_runs])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date", ascending=False, ignore_index=True)


def _history_runs(runner: Runner, horse_rows: pd.DataFrame) -> pd.DataFrame:
    if runner.horse_id is not None:
        rows = horse_rows[horse_rows["horse_id"] == runner.horse_id]
    else:
        rows = horse_rows[horse_rows["horse"] == runner.horse]
    return pd.DataFrame({c: rows[_HISTORY_RUN_COLUMNS.get(c, c)] for c in RUN_COLUMNS}).sort_values(
        "date", ascending=False, ignore_index=True
    )


def _performance(runs: pd.DataFrame) -> np.ndarray:
    """1 for a win down to 0 for last place (field of 16 assumed when unknown)."""
    field = pd.to_numeric(runs["field_size"], errors="coerce").fillna(16).clip(lower=2)
    return np.clip(1 - (runs["finish"].to_numpy(float) - 1) / (field.to_numpy() - 1), 0, 1)


def _decayed_mean(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    w = _DECAY ** np.arange(len(values))
    return float(np.sum(w * values) / w.sum())


def _known_decayed(values: pd.Series, n: int = 5) -> float:
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy()[:n]
    return _decayed_mean(v)


def runner_factors(card: RaceCard, runner: Runner) -> dict:
    """Factors that need only the runner's own card fields, not its past runs."""
    month = pd.Timestamp(card.date).month
    out = {
        "weight_carried": runner.weight_carried,
        "horse_weight_change": (
            abs(runner.horse_weight_change) if runner.horse_weight_change is not None else None
        ),
        "age": runner.age,
        "paddock_score": runner.paddock_score,
        "inbreeding": (
            inbreeding_coefficient(runner.inbreeding_crosses) if runner.inbreeding_crosses else None
        ),
    }
    if runner.weight_carried and runner.horse_weight:
        out["weight_ratio"] = runner.weight_carried / runner.horse_weight
    if runner.win_odds and runner.previous_odds:
        out["odds_drift"] = float(np.log(runner.previous_odds / runner.win_odds))
    if runner.sex is not None:
        out["summer_mare"] = float(runner.sex in FEMALE and 6 <= month <= 9)
    if runner.age is not None:
        out["autumn_three_year_old"] = float(runner.age == 3 and month >= 9)
    region = course_region(card.course)
    center = EAST_WEST.get(runner.trainer_center) if runner.trainer_center else None
    if region is not None and center is not None:
        out["long_shipping"] = float(region != center)
    return out


def horse_factors(card: RaceCard, runner: Runner, runs: pd.DataFrame) -> dict:
    out = {"n_runs": len(runs)}
    if runs.empty:
        if runner.jockey:
            out["first_ride"] = 1.0
        out["first_course"] = out["first_distance"] = 1.0
        return out
    perf = _performance(runs)
    grade_w = runs["grade"].map(_CLASS_WEIGHT).fillna(_OTHER_CLASS_WEIGHT).to_numpy()
    out["form"] = _decayed_mean(perf[:5])
    out["class_form"] = _decayed_mean((grade_w * perf)[:10])

    sims = np.array(
        [
            similarity(
                r.surface if isinstance(r.surface, str) else card.surface,
                r.distance if pd.notna(r.distance) else card.distance,
                r.going if isinstance(r.going, str) else None,
                r.course if isinstance(r.course, str) else None,
                card.surface,
                card.distance,
                card.going,
                card.course,
            )
            for r in runs.itertuples()
        ]
    )
    known_cond = runs["distance"].notna().to_numpy()
    if (sims * known_cond).sum() >= 0.3:
        out["similar_form"] = float(np.sum(sims * known_cond * perf) / np.sum(sims * known_cond))

    top3 = runs["finish"].to_numpy() <= 3
    going_known = runs["going"].map(lambda g: isinstance(g, str)).to_numpy()
    if going_known.any():
        same = going_known & np.array(
            [GOING_GROUP.get(g) == GOING_GROUP[card.going] for g in runs["going"]]
        )
        out["going_aptitude"] = float(
            shrunk_rate(top3[same].sum(), same.sum(), _BASE_TOP3, _HORSE_M)
        )
    if runs["course"].notna().any():
        same = (
            (runs["course"] == card.course)
            & (runs["surface"] == card.surface)
            & ((runs["distance"] - card.distance).abs() <= 200)
        ).to_numpy()
        out["course_aptitude"] = float(
            shrunk_rate(top3[same].sum(), same.sum(), _BASE_TOP3, _HORSE_M)
        )
    closing = runs.loc[(runs["surface"] == card.surface) & runs["last3f"].notna(), "last3f"]
    if not closing.empty:
        out["closing_speed"] = -float(closing.head(3).mean())

    last = runs.iloc[0]
    today = pd.Timestamp(card.date)
    if pd.notna(last["distance"]):
        out["distance_change"] = abs(np.log(card.distance / last["distance"]))
    days = (today - last["date"]).days
    out["layoff"] = float(days >= 90)
    out["days_since_last"] = float(np.log1p(max(days, 0)))
    out["last_race_class"] = _CLASS_WEIGHT.get(last["grade"], _OTHER_CLASS_WEIGHT)
    gaps = [days, *(-runs["date"].diff().dt.days.iloc[1:]).tolist()]  # gap before each run
    out["runs_since_layoff"] = float(next((k for k, g in enumerate(gaps) if g >= 90), len(gaps)))
    out["runs_since_layoff"] = min(out["runs_since_layoff"], 4.0)

    jockeys = runs["jockey"].dropna()
    if runner.jockey and not jockeys.empty:
        mine = (runs["jockey"] == runner.jockey).to_numpy()
        if isinstance(last["jockey"], str):
            out["jockey_change"] = float(last["jockey"] != runner.jockey)
        out["first_ride"] = float(not mine.any())
        out["jockey_horse_rate"] = float(
            shrunk_rate(top3[mine].sum(), mine.sum(), _BASE_TOP3, _HORSE_M)
        )
    if runs["course"].notna().any():
        out["first_course"] = float(
            not ((runs["course"] == card.course) & (runs["surface"] == card.surface)).any()
        )
    if runs["distance"].notna().any():
        out["first_distance"] = float(
            not (
                (runs["surface"].fillna(card.surface) == card.surface)
                & ((runs["distance"] - card.distance).abs() <= 100)
            ).any()
        )

    out["speed_index"] = _known_decayed(runs["speed_figure"])
    margin = pd.to_numeric(runs["margin"], errors="coerce").clip(upper=3.0)
    out["margin_form"] = -_known_decayed(margin)
    field = pd.to_numeric(runs["field_size"], errors="coerce")
    early = pd.to_numeric(runs["early_position"], errors="coerce")
    out["early_speed"] = _known_decayed(1 - 2 * (early - 1) / (field.fillna(16) - 1).clip(lower=1))
    out["race_level"] = _known_decayed(runs["race_level"])
    trouble = runs["trouble"].head(5).dropna()
    if not trouble.empty:
        out["recent_trouble"] = float(trouble.astype(bool).mean())
    popularity = pd.to_numeric(runs["popularity"], errors="coerce")
    finish_pct = (runs["finish"] - 1) / (field.fillna(16) - 1).clip(lower=1)
    out["market_gap"] = _known_decayed((popularity - 1) / (field.fillna(16) - 1) - finish_pct)
    cushion = pd.to_numeric(runs["cushion"], errors="coerce")
    if card.cushion is not None and cushion.notna().any():
        w = np.exp(-np.abs(cushion - card.cushion).to_numpy() / 1.0)
        w = np.where(np.isnan(w), 0.0, w)
        if w.sum() >= 0.5:
            out["cushion_fit"] = float(np.sum(w * perf) / w.sum())
    return out


def _card_stat(runner: Runner, key: str, strength: float) -> float:
    rec = runner.stats.get(key)
    if rec is None:
        return np.nan
    return float(shrunk_rate(rec.top3, rec.starts, _BASE_TOP3, strength))


def card_stat_factors(runner: Runner) -> dict:
    """Connection and jockey factors from the card's ``stats`` (no history table)."""
    s = {
        "jockey_year_rate": ("jockey_year", _ENTITY_M),
        "jockey_graded_rate": ("jockey_graded", _ENTITY_M),
        "jockey_course_rate": ("jockey_course", _ENTITY_M),
        "trainer_rate": ("trainer", _ENTITY_M),
        "trainer_graded_rate": ("trainer_graded", _ENTITY_M),
        "trainer_jockey_rate": ("trainer_jockey", _PAIR_M),
        "owner_rate": ("owner", _ENTITY_M),
        "breeder_rate": ("breeder", _ENTITY_M),
        "region_rate": ("region", _ENTITY_M),
        "sire_aptitude": ("sire_condition", _PROFILE_M),
        "damsire_aptitude": ("damsire_condition", _PROFILE_M),
        "sibling_rate": ("siblings", _PAIR_M),
        "dam_race_rate": ("dam_race", _HORSE_M),
    }
    out = {name: _card_stat(runner, key, m) for name, (key, m) in s.items()}
    out["jockey_h2h"] = runner.stats.get("jockey_h2h", np.nan)
    return out


def history_stat_factors(card: RaceCard, before: pd.DataFrame, last_finishes: list) -> list[dict]:
    """Connection, jockey and race factors for every runner from the history table.

    ``last_finishes`` holds each runner's previous finishing position (None if unknown).
    """
    today = pd.Timestamp(card.date)
    recent3 = before[before["race_date"] >= today - pd.DateOffset(years=3)]
    recent5 = before[before["race_date"] >= today - pd.DateOffset(years=5)]
    this_year = before[before["year"] == today.year]
    graded3 = recent3[recent3["grade"].isin(GRADED)]
    course = before[(before["course"] == card.course) & (before["surface"] == card.surface)]
    cond = condition_bin(card.surface, card.distance, card.going)
    p0 = float(before["top3"].mean()) if len(before) else _BASE_TOP3

    def rates(rows, col, values, m):
        rows = rows[rows[col].isin([v for v in values if v is not None])]
        return grouped_top3_rates(rows, [col], p0, m) if len(rows) else pd.Series(dtype=float)

    r = card.runners
    jockeys = [x.jockey for x in r]
    trainers = [x.trainer for x in r]
    tables = {
        "jockey_year_rate": ("jockey", rates(this_year, "jockey", jockeys, _ENTITY_M)),
        "jockey_graded_rate": ("jockey", rates(graded3, "jockey", jockeys, _ENTITY_M)),
        "jockey_course_rate": ("jockey", rates(course, "jockey", jockeys, _ENTITY_M)),
        "trainer_rate": ("trainer", rates(recent3, "trainer", trainers, _ENTITY_M)),
        "trainer_graded_rate": ("trainer", rates(graded3, "trainer", trainers, _ENTITY_M)),
        "owner_rate": ("owner", rates(recent3, "owner", [x.owner for x in r], _ENTITY_M)),
        "breeder_rate": ("breeder", rates(recent5, "breeder", [x.breeder for x in r], _ENTITY_M)),
        "region_rate": ("region", rates(recent5, "region", [x.region for x in r], _ENTITY_M)),
    }
    pairs = before[before["trainer"].isin(trainers) & before["jockey"].isin(jockeys)]
    pair_rates = grouped_top3_rates(pairs, ["trainer", "jockey"], p0, _PAIR_M) if len(pairs) else {}
    profiles = {}
    for col in ("sire", "damsire"):
        values = [getattr(x, col) for x in r if getattr(x, col) is not None]
        rows = before[before[col].isin(values)]
        profiles[col] = profile_matrix(rows, col, _PROFILE_M) if len(rows) else pd.DataFrame()
    h2h = head_to_head(recent3, "jockey", [j for j in jockeys if j is not None])

    same_track = course[course["distance"] == card.distance]
    draw_rates = rates(same_track, "draw", [x.draw for x in r], _DRAW_M)
    editions = before[before["name_key"] == race_name_key(card.name)]
    editions = editions[editions["race_date"] >= today - pd.DateOffset(years=_EDITION_YEARS)]
    reference = same_track[same_track["race_date"] >= today - pd.DateOffset(years=_REFERENCE_YEARS)]
    edition_table = _trend_table(editions, RACE_TREND_ATTRS)
    reference_table = _trend_table(reference, COURSE_TREND_ATTRS)

    dams = [x.dam for x in r if x.dam is not None]
    dam_foals = before[before["dam"].isin(dams)] if dams else before.iloc[0:0]
    dams_raced = before[before["horse"].isin(dams)] if dams else before.iloc[0:0]

    out = []
    for x, last in zip(r, last_finishes, strict=True):
        d = {}
        if x.dam is not None:
            own = (dam_foals["horse_id"] == x.horse_id) | (dam_foals["horse"] == x.horse)
            d["sibling_rate"] = top3_rate(
                dam_foals[(dam_foals["dam"] == x.dam) & ~own], p0, _PAIR_M
            )
            d["dam_race_rate"] = top3_rate(dams_raced[dams_raced["horse"] == x.dam], p0, _HORSE_M)
        for name, (attr, table) in tables.items():
            key = getattr(x, attr)
            d[name] = float(table.get(key, np.nan)) if key is not None else np.nan
        d["trainer_jockey_rate"] = (
            float(pair_rates.get((x.trainer, x.jockey), np.nan)) if len(pair_rates) else np.nan
        )
        for col, name in (("sire", "sire_aptitude"), ("damsire", "damsire_aptitude")):
            key, prof = getattr(x, col), profiles[col]
            if key is not None and key in prof.index and cond in prof.columns:
                d[name] = float(prof.at[key, cond])
        d["jockey_h2h"] = h2h.get(x.jockey, np.nan) if x.jockey else np.nan
        d["draw_bias"] = float(draw_rates.get(x.draw, np.nan)) if x.draw is not None else np.nan
        attrs = _runner_attributes(x, last)
        d["race_trend"] = _trend_score(attrs, RACE_TREND_ATTRS, edition_table)
        d["course_trend"] = _trend_score(attrs, COURSE_TREND_ATTRS, reference_table)
        out.append(d)
    return out


def last_finish_band(finish) -> str | None:
    if finish is None or pd.isna(finish):
        return None
    return "1" if finish == 1 else "2-3" if finish <= 3 else "4-5" if finish <= 5 else "6+"


def _runner_attributes(x: Runner, last_finish) -> dict:
    return {
        "draw": None if x.draw is None else str(int(x.draw)),
        "age": None if x.age is None else str(int(x.age)),
        "sex": x.sex,
        "running_style": x.running_style,
        "last_finish": last_finish_band(last_finish),
        "sire": x.sire,
    }


def _as_key(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return str(int(v)) if isinstance(v, float) else str(v)


def _trend_table(rows: pd.DataFrame, attributes) -> dict:
    """attribute -> value -> shrunk top-3 rate minus the overall rate of ``rows``."""
    if rows.empty:
        return {}
    base = float(rows["top3"].mean())
    columns = {a: rows[a] for a in attributes if a != "last_finish"}
    if "last_finish" in attributes:
        columns["last_finish"] = rows["prev_finish"].map(last_finish_band)
    table = {}
    for attr, col in columns.items():
        g = pd.DataFrame({"key": col.map(_as_key), "top3": rows["top3"]}).dropna()
        agg = g.groupby("key")["top3"].agg(["sum", "size"])
        rates = shrunk_rate(agg["sum"], agg["size"], base, _TREND_M) - base
        table[attr] = dict(zip(agg.index, map(float, rates), strict=True))
    return table


def _card_trend_table(trends: dict) -> dict:
    """The same shape as ``_trend_table``, from ``{attribute: {value: Record}}`` on a card."""
    table = {}
    for attr, values in trends.items():
        starts = sum(r.starts for r in values.values())
        if not starts:
            continue
        base = sum(r.top3 for r in values.values()) / starts
        table[attr] = {
            v: float(shrunk_rate(r.top3, r.starts, base, _TREND_M)) - base
            for v, r in values.items()
        }
    return table


def _trend_score(attrs: dict, attributes, table: dict) -> float:
    """Mean over the runner's known attributes of (rate for its value - overall rate)."""
    diffs = [
        table[a][attrs[a]]
        for a in attributes
        if attrs.get(a) is not None and a in table and attrs[a] in table[a]
    ]
    return float(np.mean(diffs)) if diffs else np.nan


def workout_scores(card: RaceCard) -> list[float]:
    """Latest workout time vs. other runners on the same course, else the A-E rating."""
    latest = [x.workouts[0] if x.workouts else None for x in card.runners]
    scores = []
    for x, w in zip(card.runners, latest, strict=True):
        score = np.nan
        if w is not None and w.time_4f is not None:
            peers = [p.time_4f for p in latest if p and p.course == w.course and p.time_4f]
            if len(peers) >= 2 and np.std(peers) > 0:
                score = -(w.time_4f - np.mean(peers)) / np.std(peers)
        if np.isnan(score) and x.workout_rating is not None:
            score = WORKOUT_RATINGS[x.workout_rating]
        scores.append(score)
    return scores


STYLE_FRONT = {"front": 1.0, "stalker": 0.5, "midfield": -0.25, "closer": -1.0}


def _same_day_runners(card: RaceCard, history: pd.DataFrame | None) -> list[list[dict]]:
    """Earlier races today on this course and surface, from the card or the history."""
    races = [
        race["runners"] for race in card.same_day_races if race["surface"] in (None, card.surface)
    ]
    if races or history is None or card.post_time is None:
        return races
    start = pd.Timestamp(f"{card.date} {card.post_time}")
    today = history[
        (history["race_date"] == pd.Timestamp(card.date))
        & (history["course"] == card.course)
        & (history["surface"] == card.surface)
        & (history["post_time"] < start)
    ]
    out = []
    for _, rows in today.groupby("race_id"):
        out.append(
            [
                {
                    "draw": None if pd.isna(r.draw) else r.draw,
                    "early_position": None if pd.isna(r.early_position) else r.early_position,
                    "finish": int(r.finish_position),
                }
                for r in rows.itertuples()
            ]
        )
    return out


def compute_factors(card: RaceCard, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Raw factor values, one row per runner in card order, plus ``n_runs``.

    ``history`` must come from ``history.prepare_history``; only rows dated before the
    race are used.
    """
    before = horse_rows = None
    if history is not None:
        # history is sorted by date, so "before the race" is a prefix
        cut = history["race_date"].searchsorted(pd.Timestamp(card.date), side="left")
        before = history.iloc[:cut]
        ids = [x.horse_id for x in card.runners if x.horse_id is not None]
        names = [x.horse for x in card.runners]
        horse_rows = before[before["horse_id"].isin(ids) | before["horse"].isin(names)].copy()
        horse_rows["speed_figure"] = speed_figures(horse_rows, before)
        horse_rows["race_level"] = race_levels(horse_rows, before)
    runs_list = []
    for x in card.runners:
        runs = _history_runs(x, horse_rows) if horse_rows is not None else pd.DataFrame()
        runs_list.append(runs if not runs.empty else _past_runs_frame(x))
    last_finishes = [None if r.empty else r["finish"].iloc[0] for r in runs_list]
    history_stats = (
        history_stat_factors(card, before, last_finishes) if before is not None else None
    )
    workouts = workout_scores(card)
    card_trends = _card_trend_table(card.trends)
    card_course_trends = _card_trend_table(card.course_trends)

    bias = estimate_bias(_same_day_runners(card, history))

    rows = []
    for i, (x, runs) in enumerate(zip(card.runners, runs_list, strict=True)):
        row = {**runner_factors(card, x), **horse_factors(card, x, runs)}
        card_stats = card_stat_factors(x)
        if history_stats is not None:
            # history first, card stats only where history has nothing
            merged = {k: v for k, v in history_stats[i].items() if not pd.isna(v)}
            row.update({**card_stats, **merged})
        else:
            row.update(card_stats)
        if pd.isna(row.get("draw_bias", np.nan)) and x.draw is not None:
            rec = card.course_draw_stats.get(str(int(x.draw)))
            if rec is not None:
                row["draw_bias"] = float(shrunk_rate(rec.top3, rec.starts, _BASE_TOP3, _DRAW_M))
        attrs = _runner_attributes(x, x.past_runs[0].finish if x.past_runs else None)
        if pd.isna(row.get("race_trend", np.nan)):
            row["race_trend"] = _trend_score(attrs, RACE_TREND_ATTRS, card_trends)
        if pd.isna(row.get("course_trend", np.nan)):
            row["course_trend"] = _trend_score(attrs, COURSE_TREND_ATTRS, card_course_trends)
        if bias is not None:
            front = row.get("early_speed")
            if front is None or pd.isna(front):
                front = STYLE_FRONT.get(x.running_style)
            row["track_bias_today"] = bias.score(inside_score(x.draw), front)
        row["workout_score"] = workouts[i]
        row["comment_score"] = x.comment_score
        row["consensus_share"] = x.consensus_share
        rows.append(row)
    df = pd.DataFrame(rows)
    for name in FACTOR_NAMES:
        if name not in df.columns:
            df[name] = np.nan
    return df[[*FACTOR_NAMES, "n_runs"]].astype(float)
