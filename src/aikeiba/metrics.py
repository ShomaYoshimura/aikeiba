"""Race-level evaluation metrics. All probabilities are per-race win probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_EPS = 1e-15


def _frame(race_ids, **cols) -> pd.DataFrame:
    return pd.DataFrame(
        {"race_id": np.asarray(race_ids), **{k: np.asarray(v) for k, v in cols.items()}}
    )


def win_log_loss(probs, race_ids, is_winner) -> float:
    """Mean over races of -log(probability assigned to the winner).

    Dead heats average over the tied winners.
    """
    f = _frame(race_ids, p=probs, y=np.asarray(is_winner, dtype=float))
    f["ll"] = -f["y"] * np.log(np.clip(f["p"], _EPS, 1.0))
    per_race = f.groupby("race_id")[["ll", "y"]].sum()
    per_race = per_race[per_race["y"] > 0]
    return float((per_race["ll"] / per_race["y"]).mean())


def win_brier(probs, race_ids, is_winner) -> float:
    """Mean over races of sum_i (p_i - y_i)^2 (multi-class Brier score)."""
    f = _frame(race_ids, p=probs, y=np.asarray(is_winner, dtype=float))
    f["se"] = (f["p"] - f["y"]) ** 2
    return float(f.groupby("race_id")["se"].sum().mean())


def top_pick_place_rate(probs, race_ids, finish_position, k: int = 3) -> float:
    """Share of races where the highest-probability runner finished in the top ``k``."""
    f = _frame(race_ids, p=probs, pos=finish_position)
    top = f.loc[f.groupby("race_id")["p"].idxmax()]
    return float((top["pos"] <= k).mean())


@dataclass(frozen=True)
class BetResult:
    n_bets: int
    stake: float
    payout: float

    @property
    def roi(self) -> float:
        """Payout / stake (1.0 = break even). NaN when nothing was bet."""
        return self.payout / self.stake if self.stake else float("nan")


def value_bet_roi(probs, win_odds, is_winner, ev_threshold: float = 1.0) -> BetResult:
    """Flat 1-unit win bets on every runner whose expected value p * odds exceeds the threshold."""
    p = np.asarray(probs, dtype=float)
    odds = np.asarray(win_odds, dtype=float)
    won = np.asarray(is_winner, dtype=bool)
    bet = p * odds > ev_threshold
    return BetResult(
        n_bets=int(bet.sum()),
        stake=float(bet.sum()),
        payout=float(odds[bet & won].sum()),
    )


def calibration_table(probs, is_winner, bins=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 1.0)) -> pd.DataFrame:
    """Predicted vs. observed win rate per probability bin."""
    f = pd.DataFrame({"p": np.asarray(probs), "y": np.asarray(is_winner, dtype=float)})
    f["bin"] = pd.cut(f["p"], bins=list(bins), include_lowest=True)
    return f.groupby("bin", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")
    )


def expected_calibration_error(probs, outcomes, bins=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 1.0)) -> float:
    """Share-weighted mean |predicted - observed| over probability bins (0 = perfect)."""
    table = calibration_table(probs, outcomes, bins)
    if table.empty:
        return float("nan")
    weights = table["n"] / table["n"].sum()
    return float((weights * (table["predicted"] - table["observed"]).abs()).sum())
