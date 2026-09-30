"""Race card: everything known about one upcoming race, as collected before post time.

A race card is a JSON file. Claude Code's ``/predict-race`` skill writes one from
public sources; ``aikeiba-simulate`` reads it. Every per-runner field except the
identifiers is optional, because what is published depends on how close the race is
(odds, for example, only exist once betting opens).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

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
RUNNING_STYLES = ("front", "stalker", "midfield", "closer")  # 逃げ, 先行, 差し, 追込
STYLE_ALIASES = {
    **{s: s for s in RUNNING_STYLES},
    "逃げ": "front",
    "先行": "stalker",
    "差し": "midfield",
    "追込": "closer",
}


@dataclass(frozen=True)
class Record:
    """Starts and top-3 finishes under some condition (going, course and distance, ...)."""

    starts: int
    top3: int

    def __post_init__(self):
        if not 0 <= self.top3 <= self.starts:
            raise ValueError(f"invalid record: top3={self.top3}, starts={self.starts}")


@dataclass(frozen=True)
class Runner:
    number: int  # horse number (馬番)
    horse: str
    draw: int | None = None  # gate (枠番); falls back to the horse number
    jockey: str | None = None
    win_odds: float | None = None
    running_style: str | None = None
    recent_finishes: tuple[int, ...] = ()  # most recent first
    jockey_win_rate: float | None = None
    going_record: Record | None = None  # record on today's going
    course_record: Record | None = None  # record at this course and distance


@dataclass(frozen=True)
class RaceCard:
    name: str
    date: str
    course: str
    surface: str  # "turf" or "dirt"
    distance: int
    going: str
    runners: tuple[Runner, ...]
    sources: tuple[str, ...] = field(default=())

    @property
    def field_size(self) -> int:
        return len(self.runners)


def _record(d: dict | None) -> Record | None:
    return None if d is None else Record(int(d["starts"]), int(d["top3"]))


def parse_race_card(data: dict) -> RaceCard:
    race = data["race"]
    going = GOING_ALIASES.get(race["going"])
    if going is None:
        raise ValueError(f"unknown going {race['going']!r}; use one of {sorted(GOING_ALIASES)}")
    runners = []
    for r in data["runners"]:
        style = r.get("running_style")
        if style is not None and style not in STYLE_ALIASES:
            raise ValueError(f"unknown running_style {style!r} for {r['horse']}")
        odds = r.get("win_odds")
        if odds is not None and odds < 1.0:
            raise ValueError(f"win_odds must be decimal odds >= 1.0 for {r['horse']}")
        runners.append(
            Runner(
                number=int(r["number"]),
                horse=r["horse"],
                draw=r.get("draw"),
                jockey=r.get("jockey"),
                win_odds=odds,
                running_style=STYLE_ALIASES[style] if style else None,
                recent_finishes=tuple(int(x) for x in r.get("recent_finishes") or ()),
                jockey_win_rate=r.get("jockey_win_rate"),
                going_record=_record(r.get("going_record")),
                course_record=_record(r.get("course_record")),
            )
        )
    numbers = [r.number for r in runners]
    if len(set(numbers)) != len(numbers):
        raise ValueError("duplicate horse numbers")
    if len(runners) < 3:
        raise ValueError("a race card needs at least three runners")
    return RaceCard(
        name=race["name"],
        date=race["date"],
        course=race["course"],
        surface=race["surface"],
        distance=int(race["distance"]),
        going=going,
        runners=tuple(runners),
        sources=tuple(data.get("sources") or ()),
    )


def load_race_card(path: str | Path) -> RaceCard:
    return parse_race_card(json.loads(Path(path).read_text(encoding="utf-8")))
