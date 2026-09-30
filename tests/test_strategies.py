import numpy as np
import pytest

from aikeiba.metrics import expected_calibration_error
from aikeiba.strategies import StrategyParams, kelly_stake, make_picks

WIN = np.array([0.40, 0.25, 0.20, 0.10, 0.05])
TOP2 = np.array([0.60, 0.45, 0.40, 0.30, 0.25])
TOP3 = np.array([0.75, 0.60, 0.62, 0.50, 0.53])


def numbers(picks):
    return [i for _, i, _, _ in picks]


def test_hit_follows_win_then_top2_then_top3():
    picks = make_picks(WIN, TOP2, TOP3, None)
    assert numbers(picks["hit"]) == [0, 1, 2]
    assert numbers(picks["balanced"]) == numbers(picks["hit"])  # no odds: same as hit
    assert picks["value"] == []


def test_balanced_skips_overrated_runners_and_value_ranks_by_ev():
    odds = np.array([1.8, 5.0, 6.0, 10.0, 30.0])  # EVs 0.72, 1.25, 1.2, 1.0, 1.5
    picks = make_picks(WIN, TOP2, TOP3, odds, StrategyParams(balanced_min_ev=0.9))
    assert numbers(picks["balanced"])[0] == 1  # favourite skipped
    assert numbers(picks["value"]) == [4, 1, 2]
    assert all(stake > 0 for _, _, _, stake in picks["value"])


def test_kelly_stake():
    assert kelly_stake(0.5, 3.0, 1.0) == pytest.approx(0.25)
    assert kelly_stake(0.2, 3.0, 1.0) == 0.0
    assert kelly_stake(0.5, 3.0, 0.25) == pytest.approx(0.0625)


def test_expected_calibration_error():
    assert expected_calibration_error([0.3] * 10, [1] * 3 + [0] * 7) == pytest.approx(0.0)
    assert expected_calibration_error([0.9] * 10, [0] * 10) == pytest.approx(0.9)
