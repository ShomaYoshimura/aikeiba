import numpy as np
import pandas as pd
import pytest

from aikeiba.conditions import condition_bin, race_name_key, similarity
from aikeiba.factors import FACTOR_NAMES, compute_factors
from aikeiba.history import card_from_history, enrich_card, prepare_history
from aikeiba.model import standardize
from aikeiba.racecard import card_to_dict, parse_race_card
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


def raw_history():
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
    return pd.DataFrame(rows)


def small_history():
    return prepare_history(raw_history())


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
    raw = compute_factors(enrich_card(card, h))
    assert list(raw.columns) == [*FACTOR_NAMES, "n_runs"]
    assert raw.loc[0, "jockey_h2h"] > raw.loc[3, "jockey_h2h"]
    # sire records count only races R0-R4 under today's condition: S1 5/5, S2 10/15, shrunk
    # toward the card path's prior (0.2, 30 pseudo-runs) like a record a collector wrote
    assert raw.loc[0, "sire_aptitude"] == pytest.approx(shrunk_rate(5, 5, 0.2, 30))
    assert raw.loc[1, "sire_aptitude"] == pytest.approx(shrunk_rate(10, 15, 0.2, 30))
    # a race on the first day has no history at all
    first = compute_factors(enrich_card(card_from_history(h[h["race_id"] == "R0"]), h))
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


def test_course_trend_uses_same_course_and_distance_only():
    raw = raw_history()
    # the same sires at another distance, where S2 always wins: must be ignored
    others = []
    for k in range(10):  # enough to flip the ranking if the distance filter were missing
        other = raw[raw["race_id"] == "R0"].copy()
        other["race_id"], other["distance"] = f"X{k}", 2400
        other["horse_id"] = other["horse_id"] + f"x{k}"
        other["finish_position"] = [4, 1, 2, 3]
        others.append(other)
    h = prepare_history(pd.concat([raw, *others]))
    card = card_from_history(h[h["race_id"] == "R5"])
    factors = compute_factors(enrich_card(card, h))
    assert factors.loc[0, "course_trend"] > factors.loc[1, "course_trend"]  # S1 at Tokyo 1200


def test_course_trends_from_card():
    card = parse_race_card(
        {
            "race": {
                "name": "t",
                "date": "2026-10-04",
                "course": "Nakayama",
                "surface": "芝",
                "distance": 1200,
                "going": "良",
                "grade": "G1",
            },
            "runners": [
                {"number": 1, "horse": "A", "sire": "S1", "running_style": "逃げ"},
                {"number": 2, "horse": "B", "sire": "S2", "running_style": "追込"},
                {"number": 3, "horse": "C"},
            ],
            "course_trends": {
                "running_style": {
                    "逃げ": {"starts": 50, "top3": 20},
                    "追込": {"starts": 50, "top3": 5},
                },
                "sire": {"S1": {"starts": 40, "top3": 12}, "S2": {"starts": 40, "top3": 6}},
            },
        }
    )
    assert set(card.course_trends["running_style"]) == {"front", "closer"}
    raw = compute_factors(card)
    assert raw.loc[0, "course_trend"] > 0 > raw.loc[1, "course_trend"]
    assert np.isnan(raw.loc[2, "course_trend"])


def test_course_trends_reject_unknown_attributes():
    base = {
        "race": {
            "name": "t",
            "date": "2026-01-01",
            "course": "c",
            "surface": "turf",
            "distance": 1600,
            "going": "良",
        },
        "runners": [{"number": i, "horse": f"H{i}"} for i in (1, 2, 3)],
    }
    with pytest.raises(ValueError, match="unknown attribute 'draw' in course_trends"):
        parse_race_card({**base, "course_trends": {"draw": {"1": {"starts": 1, "top3": 0}}}})


def test_training_and_race_day_share_one_factor_path():
    """A card filled from history, written as race-card JSON and read back like a card from
    the collector agents, gives exactly the factors training computes."""
    import json

    from aikeiba.synthetic import generate_history
    from aikeiba.train import build_examples

    h = prepare_history(
        generate_history(
            n_years=2, days_per_year=24, races_per_day=8, n_horses=800, graded_per_year=12, seed=3
        )
    )
    example = build_examples(h)[0]
    rows = h[h["race_id"] == example.race_id]
    card = enrich_card(card_from_history(rows), h)
    assert any(x.past_runs for x in card.runners) and any(x.stats for x in card.runners)
    assert card.trends or card.course_trends or card.course_draw_stats
    round_trip = parse_race_card(json.loads(json.dumps(card_to_dict(card), ensure_ascii=False)))
    np.testing.assert_array_equal(
        compute_factors(round_trip).to_numpy(), compute_factors(card).to_numpy()
    )
    np.testing.assert_array_equal(standardize(compute_factors(round_trip)), example.x)


def test_enrich_keeps_card_values_where_history_has_none():
    h = small_history()
    base = card_from_history(h[h["race_id"] == "R5"])
    from dataclasses import replace

    from aikeiba.racecard import Record

    first = replace(base.runners[0], stats={"owner": Record(50, 25)})
    card = enrich_card(replace(base, runners=(first, *base.runners[1:])), h)
    assert card.runners[0].stats["owner"] == Record(50, 25)  # no owner column in history
    assert card.runners[0].stats["jockey_year"] == Record(5, 5)  # J1 won R0-R4
    assert all(p.date < "2025-06-10" for x in card.runners for p in x.past_runs)
