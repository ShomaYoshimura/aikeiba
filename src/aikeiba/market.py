"""Where the market and the model disagree (step 7 of the workflow).

Odds are looked at only here, after the fundamental evaluation. Three views:

- **Odds ladder.** Runners sorted by odds, with the ratio between neighbours. Runners
  whose odds are close form a cluster; a large ratio is a gap. A runner the model
  rates highly that sits just after a gap is the market's blind spot the report looks
  for.
- **Expected value** = win probability x odds, at the mean and at the lower end of the
  runner's interval (``uncertainty.py``). A runner that is good value only on average
  is a weaker bet than one that stays above 1 at the lower end.
- **R9 overbet.** The market ranks the runner much higher than the fundamental model
  does (rank by fundamental win probability, without odds) and its expected value is
  below 1. R9 changes no score; it only flags the runner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EV_BANDS = (  # (lower bound, label)
    (2.0, "最大妙味"),
    (1.5, "妙味大"),
    (1.0, "妙味あり"),
    (0.7, "許容圏"),
    (-np.inf, "割高"),
)


@dataclass(frozen=True)
class MarketParams:
    gap_ratio: float = 1.4  # odds ratio between neighbours that counts as a gap
    overbet_rank_gap: int = 3  # market rank this many places above the model's rank


def ev_label(ev: float | None) -> str | None:
    if ev is None or np.isnan(ev):
        return None
    return next(label for bound, label in EV_BANDS if ev >= bound)


def odds_ladder(numbers, horses, odds, params: MarketParams | None = None) -> list[dict]:
    """Runners by odds with the ratio to the previous one and a cluster number."""
    params = params or MarketParams()
    df = pd.DataFrame({"number": numbers, "horse": horses, "odds": odds}).sort_values(
        ["odds", "number"], ignore_index=True
    )
    rows, cluster, prev = [], 1, None
    for rank, r in enumerate(df.itertuples(), start=1):
        ratio = None if prev is None else r.odds / prev
        gap = ratio is not None and ratio >= params.gap_ratio
        if gap:
            cluster += 1
        rows.append(
            {
                "popularity": rank,
                "number": int(r.number),
                "horse": r.horse,
                "odds": float(r.odds),
                "ratio_to_previous": ratio,
                "gap_before": bool(gap),
                "cluster": cluster,
            }
        )
        prev = r.odds
    return rows


def format_ladder(ladder: list[dict], marks: dict[int, str] | None = None) -> str:
    """Plain-text ladder for the report, with the gaps drawn between clusters."""
    marks = marks or {}
    lines = []
    for row in ladder:
        if row["gap_before"]:
            lines.append(f"{'':28}↓ ×{row['ratio_to_previous']:.2f} の空白")
        mark = marks.get(row["number"], "")
        lines.append(
            f"{row['popularity']:>2}人気 {row['number']:>2} {row['horse']:<14} "
            f"{row['odds']:>6.1f}  {mark}"
        )
    return "\n".join(lines)


def market_view(
    numbers,
    horses,
    odds,
    win,
    win_lo,
    fundamental_win,
    params: MarketParams | None = None,
) -> dict:
    """Odds ladder, expected values and overbet flags. ``odds`` may be None (no market)."""
    params = params or MarketParams()
    if odds is None:
        return {"available": False}
    odds = np.asarray(odds, dtype=float)
    ev = win * odds
    ev_lo = win_lo * odds
    market_rank = pd.Series(odds).rank(method="min").to_numpy(int)
    model_rank = pd.Series(-np.asarray(fundamental_win)).rank(method="min").to_numpy(int)
    runners = []
    for i, num in enumerate(numbers):
        overbet = bool(model_rank[i] - market_rank[i] >= params.overbet_rank_gap and ev[i] < 1.0)
        runners.append(
            {
                "number": int(num),
                "horse": horses[i],
                "odds": float(odds[i]),
                "popularity": int(market_rank[i]),
                "fundamental_rank": int(model_rank[i]),
                "ev": float(ev[i]),
                "ev_lo": float(ev_lo[i]),
                "ev_label": ev_label(ev[i]),
                "value_only_on_average": bool(ev[i] >= 1.0 > ev_lo[i]),
                "r9_overbet": overbet,
            }
        )
    return {
        "available": True,
        "ladder": odds_ladder(numbers, horses, odds, params),
        "runners": runners,
    }
