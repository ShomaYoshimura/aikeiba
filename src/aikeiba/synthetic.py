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


# Graded races that recur every year: (name, grade, course, surface, distance)
_GRADED_RACES = (
    ("Synthetic Cup", "G1", "Tokyo", "turf", 2000),
    ("Synthetic Sprint", "G1", "Nakayama", "turf", 1200),
    ("Synthetic Mile", "G1", "Tokyo", "turf", 1600),
    ("Synthetic Derby", "G1", "Tokyo", "turf", 2400),
    ("Synthetic Dirt Classic", "G1", "Chukyo", "dirt", 1800),
    ("Synthetic Stakes", "G2", "Kyoto", "turf", 2200),
    ("Synthetic Trophy", "G2", "Hanshin", "turf", 1400),
    ("Synthetic Handicap", "G3", "Nakayama", "turf", 1800),
    ("Synthetic Dirt Stakes", "G3", "Tokyo", "dirt", 1600),
    ("Synthetic Autumn Cup", "G3", "Hanshin", "turf", 2000),
)
_COURSES = ("Tokyo", "Nakayama", "Kyoto", "Hanshin", "Chukyo")
_DISTANCES = (1200, 1400, 1600, 1800, 2000, 2200, 2400)
_GOINGS = ("good", "yielding", "soft", "heavy")
_STYLES = ("front", "stalker", "midfield", "closer")


def generate_history(
    n_years: int = 5,
    days_per_year: int = 60,
    races_per_day: int = 12,
    n_horses: int = 3000,
    start_year: int = 2020,
    graded_per_year: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """History table with known effects of jockey, trainer, owner, breeder, sire, damsire,
    course/distance aptitude, draw and going, so the factor model can be tested.

    ``graded_per_year`` recurring graded races (``_GRADED_RACES`` first, then generated
    ones) run once a year with strong horses; the other races draw horses at random.
    Market odds see a noisy version of the true expected performance.
    """
    from aikeiba.conditions import ALL_BINS, condition_bin

    rng = np.random.default_rng(seed)
    graded = list(_GRADED_RACES[:graded_per_year])
    for i in range(len(graded), graded_per_year):
        grade = ("G1", "G2", "G3")[i % 3]
        course = _COURSES[rng.integers(len(_COURSES))]
        surface = "turf" if rng.random() < 0.75 else "dirt"
        graded.append(
            (f"Synthetic {grade} Race {i}", grade, course, surface, int(rng.choice(_DISTANCES)))
        )
    if graded_per_year > days_per_year:
        raise ValueError("graded_per_year must not exceed days_per_year")
    n_jockeys, n_trainers, n_owners, n_breeders = 80, 150, 300, 100
    n_sires, n_damsires = 40, 80
    jockey_skill = rng.normal(0, 0.35, n_jockeys)
    trainer_skill = rng.normal(0, 0.25, n_trainers)
    owner_skill = rng.normal(0, 0.1, n_owners)
    breeder_skill = rng.normal(0, 0.1, n_breeders)
    bin_index = {b: i for i, b in enumerate(ALL_BINS)}
    sire_apt = rng.normal(0, 0.3, (n_sires, len(ALL_BINS)))
    damsire_apt = rng.normal(0, 0.15, (n_damsires, len(ALL_BINS)))
    draw_effect = {c: rng.normal(0, 0.08) for c in _COURSES}  # per gate step, inside positive

    ability = rng.normal(0, 1.0, n_horses)
    sire = rng.integers(0, n_sires, n_horses)
    damsire = rng.integers(0, n_damsires, n_horses)
    trainer = rng.integers(0, n_trainers, n_horses)
    owner = rng.integers(0, n_owners, n_horses)
    breeder = rng.integers(0, n_breeders, n_horses)
    region = rng.choice(["Hidaka", "Iburi", "Imported"], n_horses, p=[0.6, 0.3, 0.1])
    n_dams = n_horses // 2
    dam = rng.integers(0, n_dams, n_horses)
    dam_quality = rng.normal(0, 0.3, n_dams)
    ability += dam_quality[dam]  # siblings share part of their ability
    trainer_center = rng.choice(["美浦", "栗東"], n_trainers)
    early_pref = rng.normal(0, 1, n_horses)  # higher = races further forward
    main_jockey = rng.integers(0, n_jockeys, n_horses)
    style = rng.integers(0, 4, n_horses)
    birth_year = start_year - rng.integers(2, 6, n_horses)
    sex = rng.choice(["M", "F", "G"], n_horses, p=[0.5, 0.4, 0.1])
    # horse-level going preference that the model can only see through past runs
    off_going_pref = rng.normal(0, 0.2, n_horses)

    rows = []
    for year in range(start_year, start_year + n_years):
        days = pd.Timestamp(f"{year}-01-05") + pd.to_timedelta(
            np.sort(rng.choice(360, days_per_year, replace=False)), unit="D"
        )
        graded_days = set(
            np.linspace(0, days_per_year - 1, len(graded)).round().astype(int).tolist()
        )
        graded_iter = iter(graded)
        for d, day in enumerate(days):
            ability += rng.normal(0, 0.02, n_horses)
            used = np.zeros(n_horses, dtype=bool)
            for r in range(races_per_day):
                if r == races_per_day - 1 and d in graded_days:
                    name, grade, course, surface, distance = next(graded_iter)
                    size = 16
                    candidates = np.flatnonzero(~used)
                    top = candidates[np.argsort(-ability[candidates])[:60]]
                    horses = rng.choice(top, size, replace=False)
                else:
                    course = _COURSES[rng.integers(len(_COURSES))]
                    surface = "turf" if rng.random() < 0.6 else "dirt"
                    distance = int(rng.choice(_DISTANCES))
                    name, grade = f"{course} R{r + 1}", "OP"
                    size = int(rng.integers(8, 17))
                    horses = rng.choice(np.flatnonzero(~used), size, replace=False)
                used[horses] = True
                going = _GOINGS[rng.choice(4, p=[0.6, 0.2, 0.12, 0.08])]
                b = bin_index[condition_bin(surface, distance, going)]
                jockeys = np.where(
                    rng.random(size) < 0.7, main_jockey[horses], rng.integers(0, n_jockeys, size)
                )
                gates = np.ceil(np.arange(1, size + 1) / (size / 8)).astype(int).clip(1, 8)
                expected = (
                    ability[horses]
                    + jockey_skill[jockeys]
                    + trainer_skill[trainer[horses]]
                    + owner_skill[owner[horses]]
                    + breeder_skill[breeder[horses]]
                    + sire_apt[sire[horses], b]
                    + damsire_apt[damsire[horses], b]
                    + draw_effect[course] * (4.5 - gates)
                    + (off_going_pref[horses] if going in ("soft", "heavy") else 0.0)
                )
                trouble = rng.random(size) < 0.08
                perf = expected - 0.8 * trouble + rng.gumbel(0, 1.0, size)
                finish = np.empty(size, dtype=int)
                finish[np.argsort(-perf)] = np.arange(1, size + 1)
                est = expected + rng.normal(0, 0.45, size)
                p = np.exp(est - est.max())
                p /= p.sum()
                odds = np.maximum(1.0, np.round((1 - TAKEOUT) / p, 1))
                prev_odds = np.maximum(1.0, np.round(odds * np.exp(rng.normal(0, 0.15, size)), 1))
                early_rank = np.empty(size, dtype=int)
                early_rank[np.argsort(-(early_pref[horses] + rng.normal(0, 0.5, size)))] = (
                    np.arange(1, size + 1)
                )
                standard = distance / 16.5 + (1.5 if surface == "dirt" else 0.0)
                day_variant = {"good": 0.0, "yielding": 0.8, "soft": 1.8, "heavy": 3.0}[going]
                times = standard + day_variant - 0.35 * perf * distance / 1600
                cushion = round(
                    float(np.clip(9.5 - 1.2 * _GOINGS.index(going) + rng.normal(0, 0.5), 6, 12)),
                    1,
                )
                race_id = f"{day:%Y%m%d}{r + 1:02d}"
                for k, h in enumerate(horses):
                    rows.append(
                        (
                            race_id,
                            day,
                            name,
                            grade,
                            course,
                            surface,
                            distance,
                            going,
                            f"H{h:05d}",
                            f"Horse {h}",
                            k + 1,
                            gates[k],
                            finish[k],
                            odds[k],
                            f"J{jockeys[k]:03d}",
                            f"T{trainer[h]:03d}",
                            f"O{owner[h]:03d}",
                            f"S{sire[h]:03d}",
                            f"D{damsire[h]:03d}",
                            f"B{breeder[h]:03d}",
                            region[h],
                            year - birth_year[h],
                            sex[h],
                            _STYLES[style[h]],
                            round(35.5 - 0.4 * perf[k] + rng.normal(0, 0.3), 1),
                            round(float(times[k]), 1),
                            int(early_rank[k]),
                            cushion,
                            int(trouble[k]),
                            f"Dam {dam[h]:05d}",
                            trainer_center[trainer[h]],
                            prev_odds[k],
                            day + pd.Timedelta(hours=10, minutes=30 * r),
                        )
                    )
    columns = [
        RACE_ID,
        RACE_DATE,
        "race_name",
        GRADE,
        "course",
        "surface",
        "distance",
        "going",
        HORSE_ID,
        "horse",
        "number",
        "draw",
        FINISH_POSITION,
        WIN_ODDS,
        "jockey",
        "trainer",
        "owner",
        "sire",
        "damsire",
        "breeder",
        "region",
        "age",
        "sex",
        "running_style",
        "last3f",
        "time",
        "early_position",
        "cushion",
        "trouble",
        "dam",
        "trainer_center",
        "prev_odds",
        POST_TIME,
    ]
    return pd.DataFrame(rows, columns=columns)
