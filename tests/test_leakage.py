import pandas as pd
import pytest

from aikeiba.leakage import LeakageError, assert_point_in_time


def frame(available_at):
    return pd.DataFrame(
        {
            "race_id": ["r1", "r2"],
            "post_time": pd.to_datetime(["2024-05-26 15:40", "2024-05-26 15:40"]),
            "available_at": pd.to_datetime(available_at),
        }
    )


def test_values_before_post_time_pass():
    assert_point_in_time(frame(["2024-05-26 14:40", "2024-05-19 15:40"]), "available_at")


def test_value_at_or_after_post_time_fails():
    with pytest.raises(LeakageError, match="r2"):
        assert_point_in_time(frame(["2024-05-26 14:40", "2024-05-26 15:40"]), "available_at")


def test_missing_values_policy():
    df = frame(["2024-05-26 14:40", None])
    assert_point_in_time(df, "available_at")
    with pytest.raises(LeakageError, match="no available_at"):
        assert_point_in_time(df, "available_at", allow_missing=False)
