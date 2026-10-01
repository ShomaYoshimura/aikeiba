import pytest

from aikeiba.schema import validate_runners


def test_synthetic_data_satisfies_contract(small_runners):
    validate_runners(small_runners)


def test_rejects_missing_winner(small_runners):
    df = small_runners.copy()
    first_race = df["race_id"].iloc[0]
    df.loc[(df["race_id"] == first_race) & (df["finish_position"] == 1), "finish_position"] = 99
    with pytest.raises(ValueError, match="no winner"):
        validate_runners(df)


def test_rejects_duplicates(small_runners):
    with pytest.raises(ValueError, match="duplicate"):
        validate_runners(small_runners.iloc[[0, 0]])
