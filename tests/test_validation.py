import pandas as pd

from aikeiba.validation import walk_forward_by_year


def test_walk_forward_trains_only_on_past_years():
    dates = pd.to_datetime(["2020-03-01", "2021-03-01", "2022-03-01", "2023-03-01"])
    folds = walk_forward_by_year(dates, range(2020, 2025))
    assert [f.test_year for f in folds] == [2021, 2022, 2023]  # 2020 has no history
    for f in folds:
        assert dates[f.train_idx].max() < dates[f.test_idx].min()
    assert list(folds[-1].train_idx) == [0, 1, 2]


def test_max_train_years_limits_window():
    dates = pd.to_datetime(["2020-03-01", "2021-03-01", "2022-03-01", "2023-03-01"])
    (fold,) = walk_forward_by_year(dates, [2023], max_train_years=1)
    assert list(fold.train_idx) == [2]
