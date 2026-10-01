import numpy as np
import pytest

from aikeiba.probability import (
    fit_temperature,
    harville_exacta,
    harville_trifecta,
    market_probs,
    normalize_by_race,
    position_matrix,
    sample_plackett_luce,
    softmax_by_race,
)

RACES = np.array(["a", "a", "a", "b", "b"])


def test_normalize_sums_to_one_per_race():
    p = normalize_by_race([1, 2, 1, 3, 1], RACES)
    np.testing.assert_allclose([p[:3].sum(), p[3:].sum()], [1.0, 1.0])


def test_softmax_is_shift_invariant_within_race():
    scores = np.array([0.1, 2.0, -1.0, 5.0, 4.0])
    shifted = scores + np.array([10, 10, 10, -3, -3])
    np.testing.assert_allclose(softmax_by_race(scores, RACES), softmax_by_race(shifted, RACES))


def test_market_probs_remove_takeout():
    true_p = np.array([0.5, 0.3, 0.2])
    odds = 0.8 / true_p  # 20% takeout
    np.testing.assert_allclose(market_probs(odds, ["r"] * 3), true_p)


def test_harville_marginals():
    p = np.array([0.5, 0.25, 0.15, 0.1])
    ex = harville_exacta(p)
    np.testing.assert_allclose(ex.sum(axis=1), p)
    tri = harville_trifecta(p)
    assert tri.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(tri.sum(axis=2), ex)


def test_plackett_luce_sampling_matches_harville():
    p = np.array([0.45, 0.3, 0.15, 0.1])
    orders = sample_plackett_luce(p, 200_000, np.random.default_rng(0))
    m = position_matrix(orders)
    np.testing.assert_allclose(m[:, 0], p, atol=0.005)
    ex = harville_exacta(p)
    freq = np.zeros_like(ex)
    np.add.at(freq, (orders[:, 0], orders[:, 1]), 1.0)
    np.testing.assert_allclose(freq / len(orders), ex, atol=0.005)


def test_position_matrix_is_doubly_stochastic():
    orders = sample_plackett_luce(np.full(5, 0.2), 1000, np.random.default_rng(1))
    m = position_matrix(orders)
    np.testing.assert_allclose(m.sum(axis=0), 1.0)
    np.testing.assert_allclose(m.sum(axis=1), 1.0)


def test_fit_temperature_recovers_generating_temperature():
    rng = np.random.default_rng(2)
    n_races, field = 3000, 10
    scores = rng.normal(size=(n_races, field))
    true_t = 0.5
    p = np.exp(scores / true_t)
    p /= p.sum(axis=1, keepdims=True)
    winners = np.array([rng.choice(field, p=row) for row in p])
    is_winner = np.zeros((n_races, field), dtype=bool)
    is_winner[np.arange(n_races), winners] = True
    race_ids = np.repeat(np.arange(n_races), field)
    t = fit_temperature(scores.ravel(), race_ids, is_winner.ravel())
    assert t == pytest.approx(true_t, rel=0.15)
