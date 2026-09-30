"""Turning model scores into win probabilities and finishing-order probabilities.

The finishing-order model is Plackett-Luce with strengths equal to the win
probabilities. Its closed form for the top positions is the Harville formula, and
Gumbel-max sampling draws exact Plackett-Luce orders for the Monte Carlo engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aikeiba.metrics import win_log_loss


def normalize_by_race(values, race_ids) -> np.ndarray:
    """Scale non-negative values so they sum to 1 within each race."""
    v = pd.Series(np.asarray(values, dtype=float))
    return (v / v.groupby(np.asarray(race_ids)).transform("sum")).to_numpy()


def softmax_by_race(scores, race_ids, temperature: float = 1.0) -> np.ndarray:
    """Softmax of ``scores / temperature`` within each race."""
    s = pd.Series(np.asarray(scores, dtype=float) / temperature)
    groups = np.asarray(race_ids)
    s = s - s.groupby(groups).transform("max")  # numerical stability
    return normalize_by_race(np.exp(s.to_numpy()), groups)


def market_probs(win_odds, race_ids) -> np.ndarray:
    """Win probabilities implied by the odds, with the takeout removed proportionally."""
    return normalize_by_race(1.0 / np.asarray(win_odds, dtype=float), race_ids)


def fit_temperature(scores, race_ids, is_winner, grid=None) -> float:
    """Temperature that minimizes win log loss of ``softmax_by_race(scores)``."""
    grid = np.geomspace(0.05, 20.0, 121) if grid is None else np.asarray(grid)
    losses = [win_log_loss(softmax_by_race(scores, race_ids, t), race_ids, is_winner) for t in grid]
    return float(grid[int(np.argmin(losses))])


def harville_exacta(p) -> np.ndarray:
    """P[i, j] = probability that runner i wins and runner j finishes second."""
    p = np.asarray(p, dtype=float)
    out = p[:, None] * p[None, :] / (1.0 - p[:, None])
    np.fill_diagonal(out, 0.0)
    return out


def harville_trifecta(p) -> np.ndarray:
    """P[i, j, k] = probability of the exact order i, j, k in the first three places."""
    p = np.asarray(p, dtype=float)
    ex = harville_exacta(p)
    rest = 1.0 - p[:, None] - p[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ex[:, :, None] * p[None, None, :] / rest[:, :, None]
    n = len(p)
    idx = np.arange(n)
    out[idx, :, idx] = 0.0
    out[:, idx, idx] = 0.0
    out[idx, idx, :] = 0.0
    return np.nan_to_num(out)


def sample_plackett_luce(p, n_sims: int, rng: np.random.Generator, shocks=None) -> np.ndarray:
    """Draw finishing orders; row s lists runner indices from first to last.

    ``shocks`` (shape ``(n_sims, n)``) is added to the log-strengths, which is how
    common shocks and per-horse uncertainty enter the simulation. With no shocks
    the draws are exact Plackett-Luce samples.
    """
    p = np.asarray(p, dtype=float)
    keys = np.log(p)[None, :] + rng.gumbel(size=(n_sims, len(p)))
    if shocks is not None:
        keys = keys + shocks
    return np.argsort(-keys, axis=1)


def position_matrix(orders: np.ndarray) -> np.ndarray:
    """M[i, k] = share of simulations in which runner i finished in position k (0-based)."""
    n_sims, n = orders.shape
    m = np.zeros((n, n))
    np.add.at(m, (orders, np.broadcast_to(np.arange(n), orders.shape)), 1.0)
    return m / n_sims
