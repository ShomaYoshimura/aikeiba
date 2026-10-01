import numpy as np
import pytest

from aikeiba.metrics import (
    calibration_table,
    top_pick_place_rate,
    value_bet_roi,
    win_brier,
    win_log_loss,
)


def test_uniform_log_loss_is_log_field_size():
    races = ["a"] * 4 + ["b"] * 8
    probs = [0.25] * 4 + [0.125] * 8
    won = [1, 0, 0, 0] + [0] * 7 + [1]
    assert win_log_loss(probs, races, won) == pytest.approx((np.log(4) + np.log(8)) / 2)


def test_dead_heat_averages_over_winners():
    assert win_log_loss([0.5, 0.25, 0.25], ["a"] * 3, [1, 1, 0]) == pytest.approx(
        (-np.log(0.5) - np.log(0.25)) / 2
    )


def test_perfect_brier_is_zero():
    assert win_brier([1, 0, 0, 1], ["a", "a", "b", "b"], [1, 0, 0, 1]) == 0.0


def test_top_pick_place_rate():
    races = ["a"] * 3 + ["b"] * 3
    probs = [0.6, 0.3, 0.1, 0.2, 0.7, 0.1]
    finish = [1, 2, 3, 3, 5, 1]  # race b: favourite finished 5th
    assert top_pick_place_rate(probs, races, finish) == 0.5


def test_value_bet_roi():
    result = value_bet_roi([0.5, 0.1, 0.4], [3.0, 5.0, 2.0], [True, False, False])
    assert result.n_bets == 1  # only 0.5 * 3.0 exceeds 1.0
    assert result.roi == pytest.approx(3.0)
    assert np.isnan(value_bet_roi([0.1], [2.0], [True]).roi)


def test_calibration_table_bins():
    table = calibration_table([0.02, 0.03, 0.5, 0.5], [0, 0, 1, 0])
    assert table["n"].sum() == 4
    assert table["observed"].iloc[-1] == 0.5
