import numpy as np

from aikeiba.backtest import run_backtest
from aikeiba.baseline import LambdaRankBaseline
from aikeiba.features import FORM_FEATURES, add_prior_form_features


def test_baseline_probabilities_sum_to_one(small_runners):
    df = add_prior_form_features(small_runners)
    model = LambdaRankBaseline(FORM_FEATURES, num_boost_round=20).fit(df)
    probs = model.predict_proba(df)
    sums = df.assign(p=probs).groupby("race_id")["p"].sum()
    np.testing.assert_allclose(sums, 1.0)
    assert model.temperature > 0


def test_backtest_reports_model_and_market(small_runners):
    report = run_backtest(small_runners, [2020], model_kwargs={"num_boost_round": 20})
    assert set(report["predictor"]) == {"model", "market"}
    assert set(report["subset"]) <= {"all", "graded"}
    assert report["log_loss"].notna().all()
