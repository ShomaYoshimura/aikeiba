"""How sure the model is of each runner's score (step 6-3 of the workflow).

The race-day simulation (``simulate.py``) handles the luck of the race (aleatoric):
Plackett-Luce noise and shared pace and track-bias shocks. This module handles the
model's own doubt (epistemic): each runner's fundamental score is drawn ``n_draws``
times with a runner-specific standard deviation ``tau``, and every draw is turned into
win and top-3 probabilities in closed form (softmax, then Harville). The spread of
those probabilities says how robust a pick is: two runners with the same mean win
probability are not equally safe if one has a much wider interval.

``tau`` grows when the score rests on less evidence:

- ``base_tau`` for everyone, plus ``unknown_form_tau / sqrt(1 + known runs)``;
- ``no_grade_tau`` for a runner that has never run at the race's grade or above
  (step 2.5: the first try at the level is the least predictable);
- ``tau_per_risk_point`` x the risk sheet's ``sigma_points`` (R1, R2, R8 flagged).

With odds the fundamental score enters the log-strength with weight ``alpha``, so its
doubt does too. The values are priors in log-strength units, not fitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aikeiba.risk import GRADE_RANK, RunnerRisk
from aikeiba.strength import CardEvaluation


@dataclass(frozen=True)
class UncertaintyParams:
    base_tau: float = 0.15
    unknown_form_tau: float = 0.4
    no_grade_tau: float = 0.15
    tau_per_risk_point: float = 0.03
    n_draws: int = 10_000
    lower_q: float = 5.0
    upper_q: float = 95.0


@dataclass
class UncertaintyResult:
    tau: np.ndarray
    win_mean: np.ndarray
    win_lo: np.ndarray
    win_hi: np.ndarray
    top3_mean: np.ndarray
    top3_lo: np.ndarray
    top3_hi: np.ndarray

    @property
    def win_width(self) -> np.ndarray:
        return self.win_hi - self.win_lo

    def robustness_order(self) -> np.ndarray:
        """Runner indices by the lower end of the win interval, best first."""
        return np.argsort(-self.win_lo, kind="stable")


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def harville_top3(p: np.ndarray) -> np.ndarray:
    """Probability of finishing in the first three under Harville, for each row of ``p``."""
    p = np.atleast_2d(p)
    n = p.shape[1]
    eye = np.eye(n, dtype=bool)
    first = p[:, :, None] * p[:, None, :] / (1.0 - p[:, :, None])  # [s, a, b]: a 1st, b 2nd
    first[:, eye] = 0.0
    second = first.sum(axis=1)
    rest = 1.0 - p[:, :, None] - p[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rest > 1e-12, first / rest, 0.0)  # [s, a, b]
    # third[i] = sum over a, b not i of ratio[a, b] * p_i
    total = ratio.sum(axis=(1, 2))[:, None]
    without_i = total - ratio.sum(axis=2) - ratio.sum(axis=1)  # drop rows a=i and cols b=i
    third = p * without_i  # ratio[i, i] is already 0
    return p + second + third


def runner_tau(
    evaluation: CardEvaluation, risks: list[RunnerRisk] | None, params: UncertaintyParams
) -> np.ndarray:
    card = evaluation.card
    tau = params.base_tau + params.unknown_form_tau / np.sqrt(1.0 + evaluation.n_runs)
    target = GRADE_RANK.get(card.grade or "")
    if target:
        tested = np.array(
            [
                any(GRADE_RANK.get(p.grade or "", 0) >= target for p in r.past_runs)
                for r in card.runners
            ]
        )
        tau = tau + params.no_grade_tau * ~tested
    if risks is not None:
        tau = tau + params.tau_per_risk_point * np.array([r.sigma_points() for r in risks])
    return tau


def score_uncertainty(
    evaluation: CardEvaluation,
    risks: list[RunnerRisk] | None = None,
    params: UncertaintyParams | None = None,
    seed: int | None = 0,
    chunk: int = 1000,
) -> UncertaintyResult:
    params = params or UncertaintyParams()
    rng = np.random.default_rng(seed)
    tau = runner_tau(evaluation, risks, params)
    scale = evaluation.weights.alpha if evaluation.log_market is not None else 1.0
    n = evaluation.card.field_size
    wins, top3s = [], []
    for start in range(0, params.n_draws, chunk):
        m = min(chunk, params.n_draws - start)
        logits = evaluation.log_s + scale * tau * rng.normal(size=(m, n))
        p = softmax_rows(logits)
        wins.append(p)
        top3s.append(harville_top3(p))
    win, top3 = np.vstack(wins), np.vstack(top3s)
    lo, hi = params.lower_q, params.upper_q
    return UncertaintyResult(
        tau,
        win.mean(axis=0),
        np.percentile(win, lo, axis=0),
        np.percentile(win, hi, axis=0),
        top3.mean(axis=0),
        np.percentile(top3, lo, axis=0),
        np.percentile(top3, hi, axis=0),
    )
