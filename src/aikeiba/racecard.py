"""Race card: everything known about one upcoming graded race before post time.

A race card is a JSON file. Claude Code's ``/predict-race`` skill writes one from
public sources; ``aikeiba-simulate`` reads it. Apart from the identifiers every field
is optional, because what is published depends on how close the race is, and a
missing value is treated as the field average.

When a history table is available, statistics are computed from it and the
``stats`` / ``trends`` fields of the card are only a fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from aikeiba.schema import GRADED

# Canonical going codes, with the Japanese terms accepted on input.
GOING_ALIASES = {
    "good": "good",
    "良": "good",
    "yielding": "yielding",
    "稍重": "yielding",
    "soft": "soft",
    "重": "soft",
    "heavy": "heavy",
    "不良": "heavy",
}
SURFACE_ALIASES = {"turf": "turf", "芝": "turf", "dirt": "dirt", "ダート": "dirt", "ダ": "dirt"}
RUNNING_STYLES = ("front", "stalker", "midfield", "closer")  # 逃げ, 先行, 差し, 追込
STYLE_ALIASES = {
    **{s: s for s in RUNNING_STYLES},
    "逃げ": "front",
    "先行": "stalker",
    "差し": "midfield",
    "追込": "closer",
}
WORKOUT_RATINGS = {"A": 2.0, "B": 1.0, "C": 0.0, "D": -1.0, "E": -2.0}

# Per-runner statistics the card may carry when no history table is available.
RECORD_STATS = (
    "jockey_year",  # this year, all races
    "jockey_graded",  # graded races, recent years
    "jockey_course",  # at this course and surface
    "trainer",
    "trainer_graded",
    "trainer_jockey",  # this trainer with this jockey
    "owner",
    "breeder",
    "region",
    "sire_condition",  # sire's progeny under today's surface / distance band / going
    "damsire_condition",
    "siblings",  # the dam's other foals (horses out of the same dam)
    "dam_race",  # the dam's own racing record
)
NUMBER_STATS = ("jockey_h2h",)
WEATHER = {"晴": "sunny", "曇": "cloudy", "小雨": "rain", "雨": "rain", "雪": "snow"}
WEATHER_CODES = ("sunny", "cloudy", "rain", "snow")
EAST_WEST = {"美浦": "east", "栗東": "west", "east": "east", "west": "west"}
EDITION_TREND_ATTRS = ("draw", "age", "sex", "running_style", "last_finish")
COURSE_TREND_ATTRS = (
    "running_style",
    "age",
    "sex",
    "last_finish",
    "sire",
)  # share of races finished ahead of the other jockeys in the field


@dataclass(frozen=True)
class Record:
    """Starts and top-3 finishes under some condition."""

    starts: int
    top3: int

    def __post_init__(self):
        if not 0 <= self.top3 <= self.starts:
            raise ValueError(f"invalid record: top3={self.top3}, starts={self.starts}")


@dataclass(frozen=True)
class PastRun:
    date: str
    finish: int
    field_size: int | None = None
    race_name: str | None = None
    course: str | None = None
    surface: str | None = None
    distance: int | None = None
    going: str | None = None
    grade: str | None = None
    jockey: str | None = None
    last3f: float | None = None  # final 600 m time in seconds
    time: float | None = None  # final time in seconds
    margin: float | None = None  # seconds behind the winner (0 for the winner)
    early_position: int | None = None  # position at the first corner (1 = leading)
    speed_figure: float | None = None  # a published speed figure, one source for all runners
    race_level: float | None = None  # opponents' later top-3 rate, if a source publishes it
    cushion: float | None = None  # JRA cushion value on that day
    trouble: bool | None = None  # bad start, blocked or other interference
    popularity: int | None = None  # betting rank (1 = favourite)
    weight_carried: float | None = None


@dataclass(frozen=True)
class Workout:
    date: str
    course: str  # e.g. "坂路", "美浦W"
    time_4f: float | None = None
    last_1f: float | None = None
    intensity: str | None = None  # 馬なり, 強め, 一杯


@dataclass(frozen=True)
class Runner:
    number: int  # horse number (馬番)
    horse: str
    horse_id: str | None = None
    draw: int | None = None  # gate (枠番); falls back to the horse number
    jockey: str | None = None
    trainer: str | None = None
    trainer_center: str | None = None  # 美浦 (east) or 栗東 (west)
    owner: str | None = None
    sire: str | None = None
    dam: str | None = None
    damsire: str | None = None
    inbreeding_crosses: tuple[str, ...] = ()  # e.g. ("4x3", "5x5")
    breeder: str | None = None
    region: str | None = None  # 産地, e.g. 日高, 胆振, or the country for imported horses
    age: int | None = None
    sex: str | None = None
    weight_carried: float | None = None
    horse_weight: float | None = None
    horse_weight_change: float | None = None
    win_odds: float | None = None
    previous_odds: float | None = None  # earlier odds (e.g. the day before) for odds_drift
    running_style: str | None = None
    past_runs: tuple[PastRun, ...] = ()  # most recent first
    workouts: tuple[Workout, ...] = ()  # most recent first
    workout_rating: str | None = None  # A-E, a published evaluation
    stable_comment: str | None = None
    comment_score: float | None = None  # -2..2, see the predict-race skill for the rubric
    consensus_share: float | None = None  # 0..1, share of public predictions marking it
    paddock_comment: str | None = None
    paddock_score: float | None = None  # -2..2, same rubric style as comment_score
    stats: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RaceCard:
    name: str
    date: str
    course: str
    surface: str
    distance: int
    going: str
    runners: tuple[Runner, ...]
    grade: str | None = None
    post_time: str | None = None  # "HH:MM", needed to use earlier races of the same day
    cushion: float | None = None  # JRA cushion value
    moisture: float | None = None  # JRA moisture content (%), finish-line figure
    weather: str | None = None  # sunny / cloudy / rain / snow (晴/曇/雨/小雨/雪 accepted)
    wind_speed: float | None = None  # m/s
    # Earlier races on the same day and course: each {"surface", "distance", "runners":
    # [{"draw", "early_position", "finish"}]}, used for today's track bias.
    same_day_races: tuple = ()
    # Past editions of this race: attribute -> value -> Record. Attributes: draw, age,
    # sex, running_style, last_finish ("1", "2-3", "4-5", "6+").
    trends: dict = field(default_factory=dict)
    # Reference races at this course, surface and distance (all classes, recent years):
    # attribute -> value -> Record. Attributes: running_style, age, sex, last_finish, sire.
    course_trends: dict = field(default_factory=dict)
    # Gate statistics for this course, surface and distance over all races: gate -> Record.
    course_draw_stats: dict = field(default_factory=dict)
    sources: tuple[str, ...] = ()

    @property
    def field_size(self) -> int:
        return len(self.runners)

    @property
    def is_graded(self) -> bool:
        return self.grade in GRADED


def _going(value, where: str):
    if value is None:
        return None
    if value not in GOING_ALIASES:
        raise ValueError(f"unknown going {value!r} in {where}; use one of {sorted(GOING_ALIASES)}")
    return GOING_ALIASES[value]


def _surface(value, where: str):
    if value is None:
        return None
    if value not in SURFACE_ALIASES:
        raise ValueError(f"unknown surface {value!r} in {where}")
    return SURFACE_ALIASES[value]


def _record(d, where: str) -> Record:
    try:
        return Record(int(d["starts"]), int(d["top3"]))
    except (KeyError, TypeError) as e:
        raise ValueError(f"{where} must be {{'starts': n, 'top3': k}}") from e


def _stats(d: dict, horse: str) -> dict:
    out = {}
    for key, value in (d or {}).items():
        if key in RECORD_STATS:
            out[key] = _record(value, f"stats.{key} of {horse}")
        elif key in NUMBER_STATS:
            out[key] = float(value)
        else:
            raise ValueError(f"unknown stat {key!r} for {horse}")
    return out


def _runner(r: dict) -> Runner:
    horse = r["horse"]
    style = r.get("running_style")
    if style is not None and style not in STYLE_ALIASES:
        raise ValueError(f"unknown running_style {style!r} for {horse}")
    odds = r.get("win_odds")
    if odds is not None and odds < 1.0:
        raise ValueError(f"win_odds must be decimal odds >= 1.0 for {horse}")
    score = r.get("comment_score")
    if score is not None and not -2 <= score <= 2:
        raise ValueError(f"comment_score must be in [-2, 2] for {horse}")
    share = r.get("consensus_share")
    if share is not None and not 0 <= share <= 1:
        raise ValueError(f"consensus_share must be in [0, 1] for {horse}")
    rating = r.get("workout_rating")
    if rating is not None and rating not in WORKOUT_RATINGS:
        raise ValueError(f"workout_rating must be one of A-E for {horse}")
    paddock = r.get("paddock_score")
    if paddock is not None and not -2 <= paddock <= 2:
        raise ValueError(f"paddock_score must be in [-2, 2] for {horse}")
    center = r.get("trainer_center")
    if center is not None and center not in EAST_WEST:
        raise ValueError(f"trainer_center must be 美浦 or 栗東 for {horse}")
    for cross in r.get("inbreeding_crosses") or ():
        if inbreeding_coefficient((cross,)) is None:
            raise ValueError(f"inbreeding cross {cross!r} for {horse} must look like '4x3'")
    past_runs = tuple(
        PastRun(
            **{
                **p,
                "going": _going(p.get("going"), f"past_runs of {horse}"),
                "surface": _surface(p.get("surface"), f"past_runs of {horse}"),
            }
        )
        for p in r.get("past_runs") or ()
    )
    return Runner(
        **{
            k: v
            for k, v in r.items()
            if k not in ("past_runs", "workouts", "stats", "running_style", "inbreeding_crosses")
        },
        running_style=STYLE_ALIASES[style] if style else None,
        inbreeding_crosses=tuple(r.get("inbreeding_crosses") or ()),
        past_runs=past_runs,
        workouts=tuple(Workout(**w) for w in r.get("workouts") or ()),
        stats=_stats(r.get("stats"), horse),
    )


def inbreeding_coefficient(crosses) -> float | None:
    """Wright-style approximation: sum over crosses "a x b" of (1/2) ** (a + b - 1).

    Returns None if a cross cannot be parsed. Each cross is counted once even when the
    ancestor appears more than twice ("4x4x5" uses all pairs).
    """
    total = 0.0
    for cross in crosses:
        parts = str(cross).lower().replace("×", "x").replace(" ", "").split("x")
        if len(parts) < 2 or not all(p.isdigit() and int(p) > 0 for p in parts):
            return None
        gens = [int(p) for p in parts]
        for i in range(len(gens)):
            for j in range(i + 1, len(gens)):
                total += 0.5 ** (gens[i] + gens[j] - 1)
    return total


def _same_day_races(races, surface_where: str) -> tuple:
    out = []
    for k, race in enumerate(races or ()):
        runners = []
        for r in race.get("runners", []):
            if "finish" not in r:
                raise ValueError(f"same_day_races[{k}] runners need a finish position")
            runners.append(
                {
                    "draw": r.get("draw"),
                    "early_position": r.get("early_position"),
                    "finish": int(r["finish"]),
                }
            )
        out.append(
            {
                "surface": _surface(race.get("surface"), f"{surface_where}[{k}]"),
                "distance": race.get("distance"),
                "runners": runners,
            }
        )
    return tuple(out)


def _trend_records(d: dict | None, where: str, allowed: tuple[str, ...]) -> dict:
    out = {}
    for attr, values in (d or {}).items():
        if attr not in allowed:
            raise ValueError(f"unknown attribute {attr!r} in {where}; use one of {allowed}")
        records = {str(v): _record(rec, f"{where}.{attr}.{v}") for v, rec in values.items()}
        if attr == "running_style":  # accept 逃げ/先行/差し/追込 as keys too
            records = {STYLE_ALIASES.get(v, v): rec for v, rec in records.items()}
        out[attr] = records
    return out


def parse_race_card(data: dict) -> RaceCard:
    race = data["race"]
    going = _going(race["going"], "race")
    runners = tuple(_runner(r) for r in data["runners"])
    numbers = [r.number for r in runners]
    if len(set(numbers)) != len(numbers):
        raise ValueError("duplicate horse numbers")
    if len(runners) < 3:
        raise ValueError("a race card needs at least three runners")
    trends = _trend_records(data.get("trends"), "trends", EDITION_TREND_ATTRS)
    course_trends = _trend_records(data.get("course_trends"), "course_trends", COURSE_TREND_ATTRS)
    weather = race.get("weather")
    if weather is not None:
        weather = WEATHER.get(weather, weather)
        if weather not in WEATHER_CODES:
            raise ValueError(f"unknown weather {race['weather']!r}")
    course_draw = {
        str(g): _record(rec, f"course_draw_stats.{g}")
        for g, rec in (data.get("course_draw_stats") or {}).items()
    }
    return RaceCard(
        name=race["name"],
        date=race["date"],
        course=race["course"],
        surface=_surface(race["surface"], "race"),
        distance=int(race["distance"]),
        going=going,
        grade=race.get("grade"),
        post_time=race.get("post_time"),
        cushion=race.get("cushion"),
        moisture=race.get("moisture"),
        weather=weather,
        wind_speed=race.get("wind_speed"),
        same_day_races=_same_day_races(data.get("same_day_races"), "same_day_races"),
        runners=runners,
        trends=trends,
        course_trends=course_trends,
        course_draw_stats=course_draw,
        sources=tuple(data.get("sources") or ()),
    )


def load_race_card(path: str | Path) -> RaceCard:
    return parse_race_card(json.loads(Path(path).read_text(encoding="utf-8")))
