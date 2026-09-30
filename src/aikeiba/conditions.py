"""Race conditions: going groups, distance bands, condition bins and similarity.

Condition bins are the axes of the entity profile vectors (sire, jockey, ...), and
the similarity kernel weights a horse's past runs by how much they resemble today's race.
"""

from __future__ import annotations

import math
import unicodedata

GOING_GROUP = {"good": "fast", "yielding": "fast", "soft": "off", "heavy": "off"}
DISTANCE_BANDS = ((1400, "sprint"), (1800, "mile"), (2200, "middle"), (10_000, "long"))
SURFACES = ("turf", "dirt")
ALL_BINS = tuple(
    f"{s}-{band}-{g}" for s in SURFACES for _, band in DISTANCE_BANDS for g in ("fast", "off")
)


def distance_band(distance: float) -> str:
    return next(name for limit, name in DISTANCE_BANDS if distance <= limit)


def condition_bin(surface: str, distance: float, going: str) -> str:
    return f"{surface}-{distance_band(distance)}-{GOING_GROUP[going]}"


def race_name_key(name: str) -> str:
    """Normalize a race name so that e.g. 天皇賞（秋） and 天皇賞(秋) match."""
    return "".join(unicodedata.normalize("NFKC", name).split())


def similarity(
    surface: str,
    distance: float,
    going: str | None,
    course: str | None,
    ref_surface: str,
    ref_distance: float,
    ref_going: str,
    ref_course: str,
) -> float:
    """Kernel in [0, 1]: 1 for an identical race, decaying with distance and other differences."""
    w = math.exp(-abs(math.log(distance / ref_distance)) / 0.15)
    if surface != ref_surface:
        w *= 0.2
    if going is not None and GOING_GROUP[going] != GOING_GROUP[ref_going]:
        w *= 0.6
    if course is not None and course != ref_course:
        w *= 0.8
    return w
