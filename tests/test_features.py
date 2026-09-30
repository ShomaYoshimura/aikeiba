import numpy as np
import pandas as pd
import pytest

from aikeiba.features import add_prior_form_features


def runners():
    # Horse X runs three times; horse Y twice. Two-horse fields.
    return pd.DataFrame(
        {
            "race_id": ["r1", "r1", "r2", "r2", "r3", "r3"],
            "post_time": pd.to_datetime(
                ["2024-01-06 12:00"] * 2 + ["2024-02-03 12:00"] * 2 + ["2024-03-02 12:00"] * 2
            ),
            "horse_id": ["X", "Y", "X", "Z", "X", "Y"],
            "finish_position": [1, 2, 2, 1, 1, 2],
        }
    )


def test_prior_features_use_only_earlier_races():
    out = add_prior_form_features(runners()).set_index(["race_id", "horse_id"])
    assert np.isnan(out.loc[("r1", "X"), "prior_win_rate"])  # debut
    assert out.loc[("r2", "X"), "prior_win_rate"] == 1.0
    assert out.loc[("r3", "X"), "prior_win_rate"] == 0.5
    assert out.loc[("r3", "X"), "prior_starts"] == 2
    assert out.loc[("r3", "X"), "last_finish_pct"] == 1.0  # lost a 2-horse race
    assert out.loc[("r3", "Y"), "days_since_last"] == pytest.approx(56)


def test_future_results_do_not_change_past_features():
    base = add_prior_form_features(runners())
    changed = runners()
    changed.loc[changed["race_id"] == "r3", "finish_position"] = [2, 1]
    after = add_prior_form_features(changed)
    early = base["race_id"] != "r3"
    pd.testing.assert_frame_equal(
        base.loc[early].drop(columns="finish_position"),
        after.loc[early].drop(columns="finish_position"),
    )
