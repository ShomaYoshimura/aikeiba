"""Three ways to turn simulated probabilities into marks (◎ ○ ▲).

- hit (的中重視): the runners most likely to finish well, regardless of price.
  ◎ = highest win probability, ○ = highest top-2 probability of the rest,
  ▲ = highest top-3 probability of the rest.
- balanced (両立): the same order, but skipping runners the market clearly overrates
  (expected value p x odds below ``balanced_min_ev``). Without odds it equals hit.
- value (回収重視): the runners with the highest expected value above ``value_min_ev``,
  with a fractional-Kelly stake. Needs odds; may pick fewer than three. Stakes are given
  only for validated weights (``stakes=True``, see ``ModelWeights.validated``): an expected
  value from weights that have not beaten the market is not a reason to bet.

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


def make_picks(
    win, top2, top3, odds, params: StrategyParams | None = None, *, stakes: bool = True
) -> dict:
    """Strategy name -> list of (mark, runner index, expected value or None, stake or None).

    ``stakes=False`` leaves every stake None (use it for unvalidated weights)."""
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
                kelly_stake(win[i], odds[i], params.kelly_fraction)
                if stakes and name == "value"
                else None,
            )
            for mark, i in zip(MARKS, idx, strict=False)
        ]
    return out


@dataclass(frozen=True)
class ArticleParams:
    """Thresholds of the report marks (step 11-B of the workflow); initial choices."""

    honmei_min_risk: float = -7.0  # ◎ needs a risk total at or above this
    honmei_candidates: int = 3  # ◎ is the most robust of this many best-rated runners
    taikou_min_ev: float = 1.0  # ○
    tanana_min_ev: float = 1.5  # ▲ (with odds)
    tanana_min_popularity: int = 4  # ▲ is a runner the market does not rank near the top
    renka_max: int = 4  # △
    renka_min_top3: float = 0.15
    chui_min_popularity: int = 8  # ☆: long odds, low risk, rated well above its odds
    chui_max_risk: float = -3.0
    chui_rank_gap: int = 4
    drop_risk: float = -12.0  # ×: risk total at or below this
    discount_risk: float = -8.0  # ×: or this, when fancied and not good value
    discount_popularity: int = 4


ARTICLE_MARKS = ("◎", "○", "▲", "△", "☆", "×")


def article_marks(
    win, top3, win_lo, win_hi, risk_total, odds=None, params: ArticleParams | None = None
) -> list[tuple[str, int]]:
    """Report marks as (mark, runner index), in mark order.

    Each mark combines three things, never one alone: the probabilities (and how robust
    they are), the risk sheet total, and, with odds, the expected value.
    """
    p = params or ArticleParams()
    win, top3 = np.asarray(win, float), np.asarray(top3, float)
    risk = np.asarray(risk_total, float)
    n = len(win)
    ev = None if odds is None else win * np.asarray(odds, float)
    pop = None if odds is None else np.argsort(np.argsort(odds, kind="stable")) + 1
    win_rank = np.argsort(np.argsort(-win, kind="stable")) + 1

    drop = risk <= p.drop_risk
    if pop is not None:
        drop |= (risk <= p.discount_risk) & (pop <= p.discount_popularity) & (ev < 1.0)
    taken: set[int] = set(np.flatnonzero(drop).tolist())
    out: list[tuple[str, int]] = []

    def free(mask=None):
        m = np.ones(n, bool) if mask is None else mask.copy()
        m[list(taken)] = False
        return m

    def best(values, mask):
        if not mask.any():
            return None
        return int(np.argmax(np.where(mask, values, -np.inf)))

    rel_width = (np.asarray(win_hi) - np.asarray(win_lo)) / np.maximum(win, 1e-9)
    cands = [i for i in np.argsort(-win) if free()[i] and risk[i] >= p.honmei_min_risk]
    cands = cands[: p.honmei_candidates]
    narrow = [i for i in cands if rel_width[i] <= np.median(rel_width)]
    honmei = narrow[0] if narrow else (cands[0] if cands else best(win, free()))
    if honmei is not None:
        out.append(("◎", int(honmei)))
        taken.add(int(honmei))

    taikou = None
    if ev is not None:
        taikou = best(top3, free(ev >= p.taikou_min_ev))
    taikou = best(top3, free()) if taikou is None else taikou
    if taikou is not None:
        out.append(("○", taikou))
        taken.add(taikou)

    tanana = None
    if ev is not None:
        tanana = best(win, free((ev >= p.tanana_min_ev) & (pop >= p.tanana_min_popularity)))
        tanana = best(top3, free(ev >= p.taikou_min_ev)) if tanana is None else tanana
    tanana = best(top3, free()) if tanana is None else tanana
    if tanana is not None:
        out.append(("▲", tanana))
        taken.add(tanana)

    for i in np.argsort(-top3):
        if sum(m == "△" for m, _ in out) >= p.renka_max:
            break
        if int(i) not in taken and top3[i] >= p.renka_min_top3:
            out.append(("△", int(i)))
            taken.add(int(i))

    if pop is not None:
        for i in np.argsort(-win):
            i = int(i)
            if (
                i not in taken
                and pop[i] >= p.chui_min_popularity
                and risk[i] >= p.chui_max_risk
                and pop[i] - win_rank[i] >= p.chui_rank_gap
            ):
                out.append(("☆", i))
                taken.add(i)
                break

    out += [("×", int(i)) for i in np.flatnonzero(drop)]
    return out
