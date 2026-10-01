"""Phase 1 baseline: LightGBM LambdaMART scores turned into win probabilities."""

from __future__ import annotations

from collections.abc import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from aikeiba.probability import fit_temperature, softmax_by_race
from aikeiba.schema import FINISH_POSITION, POST_TIME, RACE_ID

DEFAULT_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [3],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "verbosity": -1,
}


def relevance(finish_position) -> np.ndarray:
    """LambdaMART label: 3 for a win, 2 for second, 1 for third, 0 otherwise."""
    return np.clip(4 - np.asarray(finish_position), 0, 3).astype(int)


class LambdaRankBaseline:
    """Ranking model plus a softmax temperature fitted on held-out recent races.

    The temperature is fitted on the latest ``calibration_fraction`` of the training
    races using a model that has not seen them, then the booster is refit on all
    training races. Everything uses only the data passed to ``fit``.
    """

    def __init__(
        self,
        feature_cols: Sequence[str],
        params: dict | None = None,
        num_boost_round: int = 200,
        calibration_fraction: float = 0.2,
    ):
        self.feature_cols = list(feature_cols)
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.num_boost_round = num_boost_round
        self.calibration_fraction = calibration_fraction
        self.booster: lgb.Booster | None = None
        self.temperature = 1.0

    def _train(self, df: pd.DataFrame) -> lgb.Booster:
        df = df.sort_values([RACE_ID], kind="stable")
        dataset = lgb.Dataset(
            df[self.feature_cols],
            label=relevance(df[FINISH_POSITION]),
            group=df.groupby(RACE_ID, sort=False).size().to_numpy(),
        )
        return lgb.train(self.params, dataset, num_boost_round=self.num_boost_round)

    def fit(self, df: pd.DataFrame) -> LambdaRankBaseline:
        race_start = df.groupby(RACE_ID)[POST_TIME].min().sort_values()
        n_cal = int(len(race_start) * self.calibration_fraction)
        if n_cal > 0:
            cal_races = race_start.index[-n_cal:]
            is_cal = df[RACE_ID].isin(cal_races)
            booster = self._train(df[~is_cal])
            cal = df[is_cal]
            scores = booster.predict(cal[self.feature_cols])
            self.temperature = fit_temperature(scores, cal[RACE_ID], cal[FINISH_POSITION] == 1)
        self.booster = self._train(df)
        return self

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("call fit() first")
        return self.booster.predict(df[self.feature_cols])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Win probabilities that sum to 1 within each race, aligned with ``df`` rows."""
        return softmax_by_race(self.predict_scores(df), df[RACE_ID], self.temperature)
