import numpy as np
import pandas as pd
import pytest

from aikeiba.bias import estimate_bias
from aikeiba.conditions import course_region
from aikeiba.factors import _past_runs_frame, compute_factors, horse_factors, runner_factors
from aikeiba.history import prepare_history
from aikeiba.history_metrics import race_levels, speed_figures
from aikeiba.racecard import inbreeding_coefficient, parse_race_card
from aikeiba.simulate import style_scores

RACE = {
    "name": "t",
    "date": "2026-07-05",
    "course": "Tokyo",
    "surface": "芝",
    "distance": 1600,
    "going": "良",
    "grade": "G3",
}


def card(runners, **extra):
    race = {**RACE, **extra.pop("race", {})}
    return parse_race_card({"race": race, "runners": runners, **extra})


def test_inbreeding_and_regions():
    assert inbreeding_coefficient(["4x3"]) == pytest.approx(0.5**6)
    assert inbreeding_coefficient(["4×4×5"]) == pytest.approx(0.5**7 + 2 * 0.5**8)
    assert inbreeding_coefficient(["bad"]) is None
    assert course_region("東京") == "east" and course_region("Hanshin") == "west"
    assert course_region("Sapporo") is None


def test_card_validation_of_new_fields():
    runners = [{"number": i, "horse": f"H{i}"} for i in (1, 2, 3)]
    c = card(runners, race={"weather": "小雨", "cushion": 9.1, "post_time": "15:40"})
    assert c.weather == "rain" and c.cushion == 9.1
    with pytest.raises(ValueError, match="trainer_center"):
        card([{**runners[0], "trainer_center": "Osaka"}, *runners[1:]])
    with pytest.raises(ValueError, match="inbreeding cross"):
        card([{**runners[0], "inbreeding_crosses": ["three by four"]}, *runners[1:]])
    with pytest.raises(ValueError, match="paddock_score"):
        card([{**runners[0], "paddock_score": 3}, *runners[1:]])


def test_runner_factors():
    c = card(
        [
            {
                "number": 1,
                "horse": "A",
                "sex": "牝",
                "age": 3,
                "win_odds": 4.0,
                "previous_odds": 8.0,
                "trainer_center": "栗東",
                "weight_carried": 55,
                "horse_weight": 440,
                "inbreeding_crosses": ["4x3"],
            },
            {"number": 2, "horse": "B", "sex": "牡", "age": 5, "trainer_center": "美浦"},
            {"number": 3, "horse": "C"},
        ]
    )
    a, b, x = (runner_factors(c, r) for r in c.runners)
    assert a["odds_drift"] == pytest.approx(np.log(2.0))
    assert a["long_shipping"] == 1.0 and b["long_shipping"] == 0.0  # Tokyo is east
    assert a["summer_mare"] == 1.0 and b["summer_mare"] == 0.0  # July
    assert a["autumn_three_year_old"] == 0.0
    assert a["weight_ratio"] == pytest.approx(55 / 440)
    assert a["inbreeding"] == pytest.approx(0.5**6)
    assert "long_shipping" not in x and x["inbreeding"] is None


def run(date, finish, **kw):
    return {"date": date, "finish": finish, "field_size": 10, "surface": "芝", **kw}


def test_horse_factors_from_past_runs():
    runner = {
        "number": 1,
        "horse": "A",
        "jockey": "J1",
        "past_runs": [
            run(
                "2026-06-01",
                2,
                jockey="J1",
                margin=0.1,
                early_position=1,
                popularity=6,
                course="Tokyo",
                distance=1600,
                trouble=False,
                speed_figure=105,
            ),
            run(
                "2026-05-01",
                9,
                jockey="J2",
                margin=2.0,
                early_position=2,
                popularity=1,
                distance=2000,
                trouble=True,
                speed_figure=95,
            ),
            run("2025-12-01", 1, jockey="J1", distance=1600, speed_figure=100),
        ],
    }
    c = card([runner, {"number": 2, "horse": "B", "jockey": "J9"}, {"number": 3, "horse": "C"}])
    r = c.runners[0]
    f = horse_factors(c, r, _past_runs_frame(r))
    assert f["runs_since_layoff"] == 2  # break before the third-last run
    assert f["first_ride"] == 0.0 and f["jockey_change"] == 0.0
    assert f["jockey_horse_rate"] > 0.2
    assert f["first_course"] == 0.0 and f["first_distance"] == 0.0
    assert f["margin_form"] < 0 and f["early_speed"] > 0.5
    assert f["recent_trouble"] == pytest.approx(0.5)
    assert f["market_gap"] != 0
    assert 95 < f["speed_index"] < 105
    debut = horse_factors(c, c.runners[1], _past_runs_frame(c.runners[1]))
    assert debut["first_ride"] == 1.0 and debut["first_course"] == 1.0


def test_track_bias_from_same_day_races():
    inside_wins = [
        [{"draw": d, "early_position": None, "finish": d} for d in range(1, 9)] for _ in range(6)
    ]
    bias = estimate_bias(inside_wins)
    assert bias.inside > 0 and bias.n_races == 6
    assert estimate_bias([]) is None

    same_day = [{"surface": "芝", "distance": 1400, "runners": races} for races in inside_wins]
    c = card(
        [{"number": i, "horse": f"H{i}", "draw": i} for i in range(1, 9)],
        same_day_races=same_day,
    )
    raw = compute_factors(c)
    assert raw.loc[0, "track_bias_today"] > raw.loc[7, "track_bias_today"]


def history_rows(extra_race: bool = False):
    races = [("R0", "2025-01-05"), ("R1", "2025-01-05"), ("R2", "2025-02-02"), ("R3", "2025-03-02")]
    if extra_race:
        races.append(("R9", "2025-04-01"))  # R0's horses run again
    rows = []
    for r, (race_id, day) in enumerate(races):
        for k in range(4):
            horse = k if race_id in ("R0", "R1", "R9") else k + 4 * r
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": day,
                    "race_name": "x",
                    "course": "Tokyo",
                    "surface": "turf",
                    "distance": 1600,
                    "going": "good",
                    "horse_id": f"H{horse}",
                    "horse": f"Horse {horse}",
                    "finish_position": k + 1,
                    "time": 96.0 + k * 0.2 + (2.0 if race_id == "R1" else 0.0),
                    "weight_carried": 55,
                    "draw": k + 1,
                    "post_time": f"{day} {10 + r}:00",
                }
            )
    return prepare_history(pd.DataFrame(rows))


def test_speed_figures_remove_the_days_variant():
    h = history_rows()
    figs = speed_figures(h, h)
    day = h[h["race_date"] == "2025-01-05"]
    # R1 ran 2 s slower on the same day: the variant absorbs part of it, order stays
    assert figs[day.index[0]] > figs[day.index[3]]
    assert figs.notna().all()


def test_race_levels_measure_opponents_later_results():
    h = history_rows(extra_race=True)
    levels = race_levels(h[h["race_id"] == "R0"], h)
    assert levels.notna().all()
    # the winner's opponents finished 2-4 later, so its level is below the others'
    assert levels.iloc[0] < levels.iloc[3]
    assert race_levels(h[h["race_id"] == "R3"], h).isna().all()  # nobody ran again


def test_style_scores_prefer_measured_early_speed():
    c = card([{"number": i, "horse": f"H{i}", "running_style": "追込"} for i in (1, 2, 3)])
    np.testing.assert_allclose(style_scores(c), [-1, -1, -1])
    np.testing.assert_allclose(style_scores(c, np.array([0.9, np.nan, 0.1])), [0.9, -1, 0.1])
