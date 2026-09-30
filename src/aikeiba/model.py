"""Linear Plackett-Luce model over the standardized factor vectors.

log-strength_i = x_i . w  (fundamental score)

and, when the odds are known (Benter-style second stage),

log-strength_i = alpha * (x_i . w) + beta * log(market probability_i)

``w`` is fitted by maximizing the Plackett-Luce likelihood of the first ``top_k``
finishing places of past graded races, with an L2 penalty. ``alpha`` and ``beta``
are fitted on later races than ``w`` so that the market weight is not inflated by an
over-fitted fundamental score. Until a fit exists, the prior weights in
``factors.FACTORS`` are used.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from aikeiba.factors import FACTOR_NAMES, FACTORS

_CLIP = 3.0


def standardize(raw: pd.DataFrame, features: Sequence[str] = FACTOR_NAMES) -> np.ndarray:
    """Within-race z-scores; unknown values and constant factors become 0; clipped at ±3."""
    x = raw[list(features)].to_numpy(dtype=float)
    known = ~np.isnan(x)
    count = known.sum(axis=0)
    safe = np.where(known, x, 0.0)
    mean = np.divide(safe.sum(axis=0), count, out=np.zeros(x.shape[1]), where=count > 0)
    var = np.divide(
        (np.where(known, x - mean, 0.0) ** 2).sum(axis=0),
        count,
        out=np.zeros(x.shape[1]),
        where=count > 0,
    )
    sd = np.sqrt(var)
    z = np.divide(x - mean, sd, out=np.zeros_like(x), where=(sd > 1e-12) & known)
    return np.clip(np.nan_to_num(z), -_CLIP, _CLIP)


@dataclass
class ModelWeights:
    fundamental: dict[str, float]
    alpha: float = 0.3  # weight of the fundamental score next to the market
    beta: float = 1.0  # weight of log market probability
    fitted: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(self.fundamental)

    def vector(self) -> np.ndarray:
        return np.array(list(self.fundamental.values()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(vars(self), ensure_ascii=False, indent=2), "utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ModelWeights:
        data = json.loads(Path(path).read_text("utf-8"))
        unknown = set(data["fundamental"]) - set(FACTOR_NAMES)
        if unknown:
            raise ValueError(f"weights file has unknown factors: {sorted(unknown)}")
        # factors added after the file was written get weight 0
        data["fundamental"] = {n: data["fundamental"].get(n, 0.0) for n in FACTOR_NAMES}
        return cls(**data)


def prior_weights() -> ModelWeights:
    return ModelWeights({f.name: f.prior_weight for f in FACTORS})


def log_strengths(x: np.ndarray, weights: ModelWeights, log_market: np.ndarray | None = None):
    f = x @ weights.vector()
    s = weights.alpha * f + weights.beta * log_market if log_market is not None else f
    return s - s.max()


def _pad(xs: Sequence[np.ndarray], orders: Sequence[np.ndarray], top_k: int):
    n_max = max(len(x) for x in xs)
    d = xs[0].shape[1]
    X = np.zeros((len(xs), n_max, d))
    valid = np.zeros((len(xs), n_max), dtype=bool)
    order = np.zeros((len(xs), top_k), dtype=int)
    for r, (x, o) in enumerate(zip(xs, orders, strict=True)):
        X[r, : len(x)] = x
        valid[r, : len(x)] = True
        order[r] = o[:top_k]
    return X, valid, order


def fit_plackett_luce(
    xs: Sequence[np.ndarray],
    orders: Sequence[np.ndarray],
    *,
    l2: float = 1.0,
    top_k: int = 3,
    w0: np.ndarray | None = None,
) -> np.ndarray:
    """Weights maximizing the Plackett-Luce likelihood of the first ``top_k`` places.

    ``xs[r]`` is the (runners x features) matrix of race r and ``orders[r]`` lists runner
    indices from the winner down (at least ``top_k`` of them).
    """
    X, valid, order = _pad(xs, orders, top_k)
    rows = np.arange(len(xs))

    def objective(w):
        s = X @ w
        remaining = valid.copy()
        nll, grad = 0.0, np.zeros_like(w)
        for j in range(top_k):
            masked = np.where(remaining, s, -np.inf)
            m = masked.max(axis=1, keepdims=True)
            e = np.exp(masked - m)
            z = e.sum(axis=1, keepdims=True)
            p = e / z
            chosen = order[:, j]
            nll -= np.sum(s[rows, chosen] - (m[:, 0] + np.log(z[:, 0])))
            grad -= (X[rows, chosen] - np.einsum("rn,rnd->rd", p, X)).sum(axis=0)
            remaining[rows, chosen] = False
        return nll + 0.5 * l2 * w @ w, grad + l2 * w

    w0 = np.zeros(X.shape[2]) if w0 is None else w0
    return minimize(objective, w0, jac=True, method="L-BFGS-B").x


def win_probabilities(x: np.ndarray, weights: ModelWeights, log_market=None) -> np.ndarray:
    s = np.exp(log_strengths(x, weights, log_market))
    return s / s.sum()
