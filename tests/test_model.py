import numpy as np
import pytest

from aikeiba.model import ModelWeights, fit_plackett_luce, log_strengths, prior_weights
from aikeiba.train import build_examples, fit_weights, walk_forward


def test_plackett_luce_fit_recovers_true_weights():
    rng = np.random.default_rng(0)
    true_w = np.array([0.8, -0.4, 0.0])
    xs, orders = [], []
    for _ in range(1500):
        x = rng.normal(size=(12, 3))
        keys = x @ true_w + rng.gumbel(size=12)
        xs.append(x)
        orders.append(np.argsort(-keys))
    w = fit_plackett_luce(xs, orders, l2=0.1, top_k=3)
    np.testing.assert_allclose(w, true_w, atol=0.08)


def test_weights_round_trip(tmp_path):
    w = prior_weights()
    w.fundamental["form"] = 0.5
    w.save(tmp_path / "w.json")
    loaded = ModelWeights.load(tmp_path / "w.json")
    assert loaded.fundamental == w.fundamental and not loaded.fitted


def test_market_combination():
    w = prior_weights()
    x = np.zeros((3, len(w.fundamental)))
    lm = np.log(np.array([0.5, 0.3, 0.2]))
    s = log_strengths(x, w, lm)
    p = np.exp(s) / np.exp(s).sum()
    np.testing.assert_allclose(p, [0.5, 0.3, 0.2])


@pytest.fixture(scope="module")
def examples():
    from aikeiba.synthetic import generate_history

    h = generate_history(
        n_years=3, days_per_year=24, races_per_day=8, n_horses=1200, graded_per_year=24, seed=4
    )
    return build_examples(h)


def test_examples_are_graded_races_after_warmup(examples):
    assert 40 <= len(examples) <= 48  # graded races after the one-year warm-up
    e = examples[0]
    assert e.x.shape[0] == 16 and e.log_market is not None
    assert sorted(e.order) == list(range(16))


def test_fit_and_walk_forward(examples):
    w = fit_weights(examples, l2=5.0)
    assert w.fitted and w.meta["n_races"] == len(examples)
    report = walk_forward(examples, [2022], l2=5.0, min_train=10)
    assert set(report.columns) >= {"fitted", "prior", "market", "fitted+market"}
