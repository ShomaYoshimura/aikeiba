"""Fit the factor weights on past graded races and check them walk-forward.

Every past graded race becomes one training example: its factors are computed as
they would have been on race day (history before that date only) and its first
three finishers are the target. Statistics inside the factors use all races; only
the examples are restricted to graded races, the prediction target.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aikeiba.factors import FACTOR_NAMES, FACTORS, compute_factors
from aikeiba.history import card_from_history, prepare_history
from aikeiba.model import ModelWeights, fit_plackett_luce, prior_weights, standardize
from aikeiba.schema import GRADED

TOP_K = 3


@dataclass
class RaceExample:
    race_id: str
    date: pd.Timestamp
    x: np.ndarray  # standardized factors, runners x features
    order: np.ndarray  # runner indices from the winner down
    log_market: np.ndarray | None


def build_examples(
    history: pd.DataFrame, grades=GRADED, min_history_days: int = 365
) -> list[RaceExample]:
    """One example per target race, skipping the first ``min_history_days`` of history."""
    h = prepare_history(history)
    start = h["race_date"].min() + pd.Timedelta(days=min_history_days)
    targets = h[h["grade"].isin(grades) & (h["race_date"] >= start)]
    examples = []
    for race_id, rows in targets.groupby("race_id", sort=False):
        if len(rows) < TOP_K + 1:
            continue
        card = card_from_history(rows)
        x = standardize(compute_factors(card, h))
        order = np.argsort(rows["finish_position"].to_numpy(), kind="stable")
        odds = rows["win_odds"].to_numpy(dtype=float)
        log_market = None
        if not np.isnan(odds).any():
            inv = 1.0 / odds
            log_market = np.log(inv / inv.sum())
        examples.append(RaceExample(race_id, rows["race_date"].iloc[0], x, order, log_market))
    return examples


def _winner_log_loss(examples, weights: ModelWeights, use_market: bool) -> float:
    losses = []
    for e in examples:
        lm = e.log_market if use_market else None
        s = e.x @ weights.vector()
        if lm is not None:
            s = weights.alpha * s + weights.beta * lm
        s = s - s.max()
        losses.append(-(s[e.order[0]] - np.log(np.exp(s).sum())))
    return float(np.mean(losses))


def _market_log_loss(examples) -> float:
    return float(np.mean([-e.log_market[e.order[0]] for e in examples]))


def fit_weights(
    examples: list[RaceExample], l2: float = 5.0, combine_fraction: float = 0.3
) -> ModelWeights:
    """Fundamental weights on all examples; market combination on the latest share of them,
    scored by a fundamental model that has not seen those races."""
    examples = sorted(examples, key=lambda e: e.date)
    w = fit_plackett_luce([e.x for e in examples], [e.order for e in examples], l2=l2, top_k=TOP_K)
    alpha, beta = prior_weights().alpha, prior_weights().beta
    with_odds = [e for e in examples if e.log_market is not None]
    n_combine = int(len(with_odds) * combine_fraction)
    if n_combine >= 30:
        early, late = with_odds[:-n_combine], with_odds[-n_combine:]
        w_early = fit_plackett_luce(
            [e.x for e in early], [e.order for e in early], l2=l2, top_k=TOP_K
        )
        xs = [np.column_stack([e.x @ w_early, e.log_market]) for e in late]
        alpha, beta = fit_plackett_luce(xs, [e.order for e in late], l2=0.0, top_k=1)
    return ModelWeights(
        fundamental=dict(zip(FACTOR_NAMES, map(float, w), strict=True)),
        alpha=float(alpha),
        beta=float(beta),
        fitted=True,
        meta={
            "n_races": len(examples),
            "first_race": str(examples[0].date.date()),
            "last_race": str(examples[-1].date.date()),
            "l2": l2,
            "top_k": TOP_K,
        },
    )


def walk_forward(
    examples: list[RaceExample], test_years, l2: float = 5.0, min_train: int = 30
) -> pd.DataFrame:
    """Winner log loss per test year: fitted vs. prior weights, with and without the market."""
    rows = []
    prior = prior_weights()
    for year in test_years:
        train = [e for e in examples if e.date.year < year]
        test = [e for e in examples if e.date.year == year]
        if len(train) < min_train or not test:
            continue
        fitted = fit_weights(train, l2=l2)
        row = {
            "test_year": year,
            "n_train": len(train),
            "n_test": len(test),
            "fitted": _winner_log_loss(test, fitted, use_market=False),
            "prior": _winner_log_loss(test, prior, use_market=False),
            "uniform": float(np.mean([np.log(len(e.x)) for e in test])),
        }
        with_odds = [e for e in test if e.log_market is not None]
        if with_odds:
            row["fitted+market"] = _winner_log_loss(with_odds, fitted, use_market=True)
            row["market"] = _market_log_loss(with_odds)
        rows.append(row)
    return pd.DataFrame(rows)


def weights_table(weights: ModelWeights) -> pd.DataFrame:
    groups = {f.name: f.group for f in FACTORS}
    return (
        pd.DataFrame(
            {
                "factor": list(weights.fundamental),
                "group": [groups[n] for n in weights.fundamental],
                "weight": list(weights.fundamental.values()),
            }
        )
        .assign(abs_weight=lambda d: d["weight"].abs())
        .sort_values("abs_weight", ascending=False)
        .drop(columns="abs_weight")
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", help="history parquet; omit to use synthetic history")
    parser.add_argument("--out", default="data/models/weights.json")
    parser.add_argument("--test-years", type=int, nargs="*")
    parser.add_argument("--l2", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0, help="synthetic history seed")
    args = parser.parse_args(argv)

    if args.history:
        history = pd.read_parquet(args.history)
    else:
        from aikeiba.synthetic import generate_history

        history = generate_history(seed=args.seed)
    examples = build_examples(history)
    years = sorted({e.date.year for e in examples})
    test_years = args.test_years if args.test_years is not None else years[1:]
    with pd.option_context("display.width", 120, "display.float_format", "{:.4f}".format):
        print(f"{len(examples)} graded races as examples\n")
        report = walk_forward(examples, test_years, l2=args.l2)
        if not report.empty:
            print("Walk-forward winner log loss (lower is better):")
            print(report.to_string(index=False), "\n")
        weights = fit_weights(examples, l2=args.l2)
        print(f"market combination: alpha={weights.alpha:.3f} beta={weights.beta:.3f}")
        print(weights_table(weights).to_string(index=False))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    weights.save(args.out)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
