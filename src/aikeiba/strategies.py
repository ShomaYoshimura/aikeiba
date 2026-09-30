"""Three ways to turn simulated probabilities into marks (◎ ○ ▲).

- hit (的中重視): the runners most likely to finish well, regardless of price.
  ◎ = highest win probability, ○ = highest top-2 probability of the rest,
  ▲ = highest top-3 probability of the rest.
- balanced (両立): the same order, but skipping runners the market clearly overrates
  (expected value p x odds below ``balanced_min_ev``). Without odds it equals hit.
- value (回収重視): the runners with the highest expected value above ``value_min_ev``,
  with a fractional-Kelly stake. Needs odds; may pick fewer than three.

The thresholds are initial choices; tune them on the prediction log.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STRATEGIES = ("hit", "balanced", "value")
MARKS = ("◎", "○", "▲")


@dataclass(frozen=True)
class StrategyParams:
    balanced_min_ev: float = 0.9
    value_min_ev: float = 1.1
    kelly_fraction: float = 0.25


def expected_values(win: np.ndarray, odds: np.ndarray | None) -> np.ndarray | None:
    return None if odds is None else win * odds


def kelly_stake(p: float, odds: float, fraction: float) -> float:
    """Fraction of the bankroll for a win bet: fractional Kelly, never negative."""
    if odds <= 1.0:
        return 0.0
    return max(0.0, (p * odds - 1.0) / (odds - 1.0)) * fraction


def _ordered_picks(win, top2, top3, eligible: np.ndarray) -> list[int]:
    chosen: list[int] = []
    for p in (win, top2, top3):
        masked = np.where(eligible, p, -1.0)
        masked[chosen] = -1.0
        if masked.max() < 0:  # not enough eligible runners: fall back to everyone
            masked = p.copy()
            masked[chosen] = -1.0
        chosen.append(int(np.argmax(masked)))
    return chosen


def hit_picks(win, top2, top3) -> list[int]:
    return _ordered_picks(win, top2, top3, np.ones(len(win), dtype=bool))


def balanced_picks(win, top2, top3, ev, min_ev: float) -> list[int]:
    eligible = np.ones(len(win), dtype=bool) if ev is None else ev >= min_ev
    return _ordered_picks(win, top2, top3, eligible)


def value_picks(ev, min_ev: float) -> list[int]:
    if ev is None:
        return []
    candidates = [int(i) for i in np.argsort(-ev) if ev[i] > min_ev]
    return candidates[:3]


def make_picks(win, top2, top3, odds, params: StrategyParams | None = None) -> dict:
    """Strategy name -> list of (mark, runner index, expected value or None, stake or None)."""
    params = params or StrategyParams()
    ev = expected_values(win, odds)
    out = {}
    for name, idx in (
        ("hit", hit_picks(win, top2, top3)),
        ("balanced", balanced_picks(win, top2, top3, ev, params.balanced_min_ev)),
        ("value", value_picks(ev, params.value_min_ev)),
    ):
        out[name] = [
            (
                mark,
                i,
                None if ev is None else float(ev[i]),
                kelly_stake(win[i], odds[i], params.kelly_fraction) if name == "value" else None,
            )
            for mark, i in zip(MARKS, idx, strict=False)
        ]
    return out
