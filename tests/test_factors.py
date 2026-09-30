import numpy as np
import pandas as pd
import pytest

from aikeiba.conditions import condition_bin, race_name_key, similarity
from aikeiba.factors import FACTOR_NAMES, compute_factors
from aikeiba.history import card_from_history, prepare_history
from aikeiba.model import standardize
from aikeiba.racecard import parse_race_card
from aikeiba.stats import head_to_head, profile_matrix, shrunk_rate


def test_condition_helpers():
    assert condition_bin("turf", 1200, "yielding") == "turf-sprint-fast"
    assert condition_bin("dirt", 2400, "heavy") == "dirt-long-off"
    assert race_name_key("天皇賞（秋）") == race_name_key("天皇賞(秋)")
    same = similarity("turf", 2000, "good", "Tokyo", "turf", 2000, "good", "Tokyo")
    other = similarity("dirt", 1200, "heavy", "Kyoto", "turf", 2000, "good", "Tokyo")
    assert same == 1.0 and other < 0.01


def test_shrunk_rate_pulls_small_samples_to_prior():
    assert shrunk_rate(1, 1, 0.2, 9) == pytest.approx(0.28)
    assert shrunk_rate(500, 1000, 0.2, 9) == pytest.approx(0.497, abs=1e-3)


def small_history():
    rows = []
    # jockey J1 always beats J2; sire S1 strong on turf sprints
    for r in range(6):
        for k, (j, s) in enumerate([("J1", "S1"), ("J2", "S2"), ("J3", "S2"), ("J4", "S2")]):
            rows.append(
                {
                    "race_id": f"R{r}",
                    "race_date": f"2025-0{r + 1}-10",
                    "race_name": "Test S",
                    "grade": "G3",
                    "course": "Tokyo",
                    "surface": "turf",
                    "distance": 1200,
                    "going": "good",
                    "horse_id": f"H{r}{k}",
                    "horse": f"Horse {r}{k}",
                    "number": k + 1,
                    "draw": k + 1,
                    "finish_position": k + 1,
                    "jockey": j,
                    "trainer": "T1",
                    "sire": s,
                    "age": 4,
                }
            )
    return prepare_history(pd.DataFrame(rows))


def test_history_prep_and_entity_stats():
    h = small_history()
    assert {"top3", "cond_bin", "field_size", "prev_finish"} <= set(h.columns)
    prof = profile_matrix(h, "sire")
    assert prof.loc["S1", "turf-sprint-fast"] > prof.loc["S2", "turf-sprint-fast"]
    h2h = head_to_head(h, "jockey", ["J1", "J2", "J4"])
    assert h2h["J1"] > 0.5 > h2h["J4"]


def test_factors_from_history_use_only_earlier_races():
    h = small_history()
    card = card_from_history(h[h["race_id"] == "R5"])
    raw = compute_factors(card, h)
    assert list(raw.columns) == [*FACTOR_NAMES, "n_runs"]
    assert raw.loc[0, "jockey_h2h"] > raw.loc[3, "jockey_h2h"]
    assert raw.loc[0, "sire_aptitude"] > raw.loc[1, "sire_aptitude"]
    # a race on the first day has no history at all
    first = compute_factors(card_from_history(h[h["race_id"] == "R0"]), h)
    assert first["jockey_h2h"].isna().all()


def test_factors_from_card_only():
    card = parse_race_card(
        {
            "race": {
                "name": "t",
                "date": "2026-10-04",
                "course": "Tokyo",
                "surface": "芝",
                "distance": 1600,
                "going": "良",
                "grade": "G2",
            },
            "runners": [
                {
                    "number": 1,
                    "horse": "A",
                    "stats": {"jockey_year": {"starts": 100, "top3": 40}},
                    "past_runs": [{"date": "2026-09-01", "finish": 1, "grade": "G1"}],
                    "comment_score": 2,
                },
                {
                    "number": 2,
                    "horse": "B",
                    "stats": {"jockey_year": {"starts": 100, "top3": 10}},
                    "past_runs": [{"date": "2026-01-01", "finish": 10}],
                },
                {"number": 3, "horse": "C"},
            ],
        }
    )
    raw = compute_factors(card)
    assert raw.loc[0, "jockey_year_rate"] > raw.loc[1, "jockey_year_rate"]
    assert raw.loc[0, "class_form"] > raw.loc[1, "class_form"]
    assert raw.loc[1, "layoff"] == 1.0 and raw.loc[0, "layoff"] == 0.0
    assert np.isnan(raw.loc[2, "form"]) and raw.loc[2, "n_runs"] == 0


def test_standardize_treats_unknown_as_average():
    raw = pd.DataFrame({name: [np.nan, np.nan, np.nan] for name in FACTOR_NAMES})
    raw["form"] = [1.0, np.nan, 0.0]
    x = standardize(raw)
    form = list(FACTOR_NAMES).index("form")
    np.testing.assert_allclose(x[:, form], [1.0, 0.0, -1.0])
    assert np.count_nonzero(np.delete(x, form, axis=1)) == 0


def test_card_validation_errors():
    base = {
        "race": {
            "name": "t",
            "date": "2026-01-01",
            "course": "c",
            "surface": "turf",
            "distance": 1600,
            "going": "良",
        }
    }
    runners = [{"number": i, "horse": f"H{i}"} for i in (1, 2, 3)]
    with pytest.raises(ValueError, match="unknown stat"):
        parse_race_card({**base, "runners": [{**runners[0], "stats": {"typo": 1}}, *runners[1:]]})
    with pytest.raises(ValueError, match="comment_score"):
        parse_race_card({**base, "runners": [{**runners[0], "comment_score": 5}, *runners[1:]]})
