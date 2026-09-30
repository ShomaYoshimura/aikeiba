"""Log-strengths for the runners of one race card.

The strength of runner i is a weighted sum of within-race z-scores, plus the log of
the market-implied probability when odds are available. The market already prices
most public information, so the other factors get small weights when odds exist.

The weights are provisional priors, not fitted values. Replace them with values
fitted by the walk-forward backtest once real data is connected (ROADMAP Phase 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aikeiba.racecard import RaceCard, Record


@dataclass(frozen=True)
class StrengthWeights:
    market: float = 1.0
    form: float = 0.15
    jockey: float = 0.10
    going: float = 0.10
    course: float = 0.10


WITH_MARKET = StrengthWeights()
WITHOUT_MARKET = StrengthWeights(market=0.0, form=0.6, jockey=0.25, going=0.2, course=0.2)

_FORM_DECAY = 0.8  # weight of each older run relative to the next more recent one
_RECORD_PRIOR_RATE = 0.3  # shrink small samples toward this top-3 rate
_RECORD_PRIOR_STARTS = 3.0


def _zscore(x: np.ndarray) -> np.ndarray:
    """Z-score over the known values; unknown (NaN) values become 0 (field average)."""
    known = ~np.isnan(x)
    out = np.zeros_like(x)
    if known.sum() >= 2 and np.nanstd(x) > 0:
        out[known] = (x[known] - np.nanmean(x)) / np.nanstd(x)
    return out


def form_score(recent_finishes: tuple[int, ...], window: int = 5) -> float:
    """Decay-weighted mean of -log(finish position) over the last ``window`` runs."""
    runs = recent_finishes[:window]
    if not runs:
        return np.nan
    w = _FORM_DECAY ** np.arange(len(runs))
    return float(np.sum(w * -np.log(runs)) / w.sum())


def record_score(record: Record | None) -> float:
    """Top-3 rate shrunk toward the prior, so one lucky start does not dominate."""
    if record is None:
        return np.nan
    return (record.top3 + _RECORD_PRIOR_RATE * _RECORD_PRIOR_STARTS) / (
        record.starts + _RECORD_PRIOR_STARTS
    )


def market_probabilities(card: RaceCard) -> np.ndarray | None:
    """Normalized odds inverses, or None unless every runner has odds."""
    odds = [r.win_odds for r in card.runners]
    if any(o is None for o in odds):
        return None
    inv = 1.0 / np.asarray(odds, dtype=float)
    return inv / inv.sum()


def log_strengths(card: RaceCard, weights: StrengthWeights | None = None) -> np.ndarray:
    runners = card.runners
    p_mkt = market_probabilities(card)
    if weights is None:
        weights = WITH_MARKET if p_mkt is not None else WITHOUT_MARKET
    rate = [np.nan if r.jockey_win_rate is None else r.jockey_win_rate for r in runners]
    s = (
        weights.form * _zscore(np.array([form_score(r.recent_finishes) for r in runners]))
        + weights.jockey * _zscore(np.array(rate, dtype=float))
        + weights.going * _zscore(np.array([record_score(r.going_record) for r in runners]))
        + weights.course * _zscore(np.array([record_score(r.course_record) for r in runners]))
    )
    if p_mkt is not None and weights.market:
        s = s + weights.market * np.log(p_mkt)
    return s - s.max()


def runs_known(card: RaceCard) -> np.ndarray:
    return np.array([len(r.recent_finishes) for r in card.runners])
