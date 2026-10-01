"""Today's track bias, estimated from earlier races on the same day, course and surface.

Each earlier runner contributes (inside, front, performance), all scaled to [-1, 1]:

- inside: 1 for gate 1 down to -1 for gate 8
- front: 1 for leading at the first corner down to -1 for last
- performance: 1 for the winner down to -1 for last

Within each race the values are centred, and a ridge regression gives two coefficients:
how much being inside and being forward helped today. The ridge penalty acts like a
number of neutral pseudo-observations, so one or two races cannot produce a strong bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_RIDGE = 30.0


@dataclass(frozen=True)
class TrackBias:
    inside: float  # performance gain per unit of "inside" (gate 1 = +1, gate 8 = -1)
    front: float  # performance gain per unit of "front" (leader = +1, last = -1)
    n_races: int

    def score(self, inside: float | None, front: float | None) -> float:
        return self.inside * (inside or 0.0) + self.front * (front or 0.0)


def inside_score(draw) -> float | None:
    return None if draw is None else 1.0 - 2.0 * (float(draw) - 1.0) / 7.0


def front_score(early_position, field_size) -> float | None:
    if early_position is None or field_size is None or field_size < 2:
        return None
    return 1.0 - 2.0 * (float(early_position) - 1.0) / (field_size - 1.0)


def estimate_bias(races: list[list[dict]]) -> TrackBias | None:
    """``races``: per race, runners as {"draw", "early_position", "finish"} (None allowed)."""
    xs, ys = [], []
    used = 0
    for runners in races:
        n = len(runners)
        rows = []
        for r in runners:
            ins = inside_score(r.get("draw"))
            fr = front_score(r.get("early_position"), n)
            if ins is None and fr is None:
                continue
            perf = 1.0 - 2.0 * (r["finish"] - 1.0) / (n - 1.0) if n > 1 else 0.0
            rows.append((ins, fr, perf))
        if len(rows) < 3:
            continue
        a = np.array([[np.nan if v is None else v for v in row] for row in rows])
        known = ~np.isnan(a)
        counts = known.sum(axis=0)
        means = np.divide(
            np.where(known, a, 0.0).sum(axis=0), counts, out=np.zeros(a.shape[1]), where=counts > 0
        )
        a = a - means  # centre within the race
        xs.append(np.nan_to_num(a[:, :2]))
        ys.append(np.nan_to_num(a[:, 2]))
        used += 1
    if not used:
        return None
    x, y = np.vstack(xs), np.concatenate(ys)
    coef = np.linalg.solve(x.T @ x + _RIDGE * np.eye(2), x.T @ y)
    return TrackBias(float(coef[0]), float(coef[1]), used)
