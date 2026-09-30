"""Walk-forward backtest of the baseline against the market."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from aikeiba.baseline import LambdaRankBaseline
from aikeiba.features import FORM_FEATURES, add_prior_form_features
from aikeiba.metrics import top_pick_place_rate, value_bet_roi, win_brier, win_log_loss
from aikeiba.probability import market_probs
from aikeiba.schema import (
    FINISH_POSITION,
    RACE_DATE,
    RACE_ID,
    WIN_ODDS,
    is_graded,
    validate_runners,
)
from aikeiba.synthetic import generate_runners
from aikeiba.validation import walk_forward_by_year


def score(df: pd.DataFrame, probs: np.ndarray, ev_threshold: float) -> dict:
    won = (df[FINISH_POSITION] == 1).to_numpy()
    bets = value_bet_roi(probs, df[WIN_ODDS], won, ev_threshold)
    return {
        "log_loss": win_log_loss(probs, df[RACE_ID], won),
        "brier": win_brier(probs, df[RACE_ID], won),
        "top_pick_top3": top_pick_place_rate(probs, df[RACE_ID], df[FINISH_POSITION]),
        "roi": bets.roi,
        "n_bets": bets.n_bets,
    }


def run_backtest(
    df: pd.DataFrame,
    test_years,
    feature_cols=FORM_FEATURES,
    ev_threshold: float = 1.1,
    model_kwargs: dict | None = None,
) -> pd.DataFrame:
    """One row per (test year, subset, predictor) with the race-level metrics."""
    validate_runners(df)
    df = add_prior_form_features(df).reset_index(drop=True)
    rows = []
    for fold in walk_forward_by_year(df[RACE_DATE], test_years):
        train, test = df.iloc[fold.train_idx], df.iloc[fold.test_idx]
        model = LambdaRankBaseline(feature_cols, **(model_kwargs or {})).fit(train)
        predictions = {
            "model": model.predict_proba(test),
            "market": market_probs(test[WIN_ODDS], test[RACE_ID]),
        }
        subsets = {"all": np.ones(len(test), dtype=bool), "graded": is_graded(test).to_numpy()}
        for subset, mask in subsets.items():
            if not mask.any():
                continue
            for name, probs in predictions.items():
                rows.append(
                    {
                        "test_year": fold.test_year,
                        "subset": subset,
                        "predictor": name,
                        "n_races": test.loc[mask, RACE_ID].nunique(),
                        **score(test[mask], probs[mask], ev_threshold),
                    }
                )
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="runner-table parquet; omit to use synthetic data")
    parser.add_argument("--test-years", type=int, nargs="+", help="years to evaluate")
    parser.add_argument("--ev-threshold", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=0, help="synthetic data seed")
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.data) if args.data else generate_runners(seed=args.seed)
    years = sorted(pd.to_datetime(df[RACE_DATE]).dt.year.unique())
    test_years = args.test_years or years[1:]
    report = run_backtest(df, test_years, ev_threshold=args.ev_threshold)
    with pd.option_context("display.width", 120, "display.float_format", "{:.4f}".format):
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()
