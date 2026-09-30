"""Log-strengths for the runners of one race card, with the reasons behind them.

The card (and the history table, when given) is abstracted into one standardized
factor vector per runner (``factors``), which the linear model (``model``) turns into
log-strengths. With odds on every runner, the market log-probability is combined in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aikeiba.factors import FACTOR_NAMES, compute_factors
from aikeiba.history import prepare_history
from aikeiba.model import ModelWeights, prior_weights, standardize
from aikeiba.model import log_strengths as model_log_strengths
from aikeiba.racecard import RaceCard


def market_log_probabilities(card: RaceCard) -> np.ndarray | None:
    """Log of normalized odds inverses, or None unless every runner has odds."""
    odds = [r.win_odds for r in card.runners]
    if any(o is None for o in odds):
        return None
    inv = 1.0 / np.asarray(odds, dtype=float)
    return np.log(inv / inv.sum())


@dataclass
class CardEvaluation:
    card: RaceCard
    raw: pd.DataFrame  # raw factor values (NaN = unknown) plus n_runs
    x: np.ndarray  # standardized factors
    weights: ModelWeights
    log_market: np.ndarray | None
    log_s: np.ndarray

    @property
    def n_runs(self) -> np.ndarray:
        return self.raw["n_runs"].to_numpy()

    def contributions(self) -> pd.DataFrame:
        """Runner x factor contributions to the fundamental score (x_ij * w_j)."""
        return pd.DataFrame(self.x * self.weights.vector(), columns=list(FACTOR_NAMES))

    def reasons(self, i: int, k: int = 3) -> list[tuple[str, float]]:
        """The ``k`` factors that move runner ``i`` most, strongest first."""
        c = self.contributions().iloc[i]
        top = c.abs().sort_values(ascending=False).head(k).index
        return [(name, float(c[name])) for name in top if abs(c[name]) > 1e-9]

    def missing_factors(self) -> list[str]:
        """Factors unknown for every runner (they contribute nothing)."""
        return [n for n in FACTOR_NAMES if self.raw[n].isna().all()]


def evaluate_card(
    card: RaceCard, history: pd.DataFrame | None = None, weights: ModelWeights | None = None
) -> CardEvaluation:
    if history is not None and "cond_bin" not in history.columns:
        history = prepare_history(history)
    weights = weights or prior_weights()
    raw = compute_factors(card, history)
    x = standardize(raw)
    log_market = market_log_probabilities(card)
    log_s = model_log_strengths(x, weights, log_market)
    return CardEvaluation(card, raw, x, weights, log_market, log_s)


def log_strengths(card: RaceCard, history=None, weights=None) -> np.ndarray:
    return evaluate_card(card, history, weights).log_s
