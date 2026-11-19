import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from aikeiba.market import ev_label, market_view, odds_ladder
from aikeiba.probability import harville_trifecta
from aikeiba.racecard import load_race_card, parse_race_card
from aikeiba.report import analyze, main, markdown_to_html, render_html
from aikeiba.risk import ITEMS, RunnerRisk, assess_risks
from aikeiba.simulate import SCENARIOS, ShockParams, classify_scenarios, simulate_race
from aikeiba.strategies import article_marks
from aikeiba.strength import evaluate_card
from aikeiba.tickets import build_tickets, evaluate_tickets, payout, return_scenarios
from aikeiba.uncertainty import UncertaintyParams, harville_top3, score_uncertainty

EXAMPLE = Path(__file__).parent.parent / "examples" / "race_card.example.json"


@pytest.fixture
def card():
    return load_race_card(EXAMPLE)


def small_card(runners=None, **race):
    base = {
        "name": "t",
        "grade": "G1",
        "date": "2026-06-01",
        "course": "Tokyo",
        "surface": "turf",
        "distance": 2000,
        "going": "良",
    }
    runners = runners or [{"number": i, "horse": f"H{i}"} for i in range(1, 5)]
    return {"race": {**base, **race}, "runners": runners}


# ----------------------------------------------------------------- race card


def test_card_parses_rotation_trends_and_past_editions(card):
    assert card.rotation_trends["Prep 0"]["6+"].starts == 40
    assert [e["year"] for e in card.past_editions] == [2025, 2024, 2023]


def test_card_rejects_bad_rotation_band_and_edition_field():
    with pytest.raises(ValueError, match="band"):
        parse_race_card(
            {**small_card(), "rotation_trends": {"X": {"7+": {"starts": 1, "top3": 0}}}}
        )
    with pytest.raises(ValueError, match="unknown fields"):
        parse_race_card({**small_card(), "past_editions": [{"year": 2025, "lap": 1}]})


# ----------------------------------------------------------------- risk sheet


def test_every_runner_gets_every_item(card):
    risks = assess_risks(card)
    assert len(risks) == card.field_size
    for r in risks:
        assert set(r.items) == set(ITEMS)
        assert set(r.notes) == set(ITEMS)
        assert all(v is None or v <= 0 for v in r.items.values())


def _runner(number, last_race, finish, **extra):
    run = {"date": "2026-05-01", "finish": finish, "race_name": last_race, "distance": 2000}
    return {"number": number, "horse": f"H{number}", "past_runs": [run], **extra}


def test_r1_rotation_thresholds():
    runners = [_runner(i, "Prep", f) for i, f in ((1, 1), (2, 2), (3, 4), (4, 9))]
    data = small_card(runners)
    data["rotation_trends"] = {
        "Prep": {
            "1": {"starts": 10, "top3": 5},  # 50% -> 0
            "2-3": {"starts": 10, "top3": 2},  # 20% -> -2
            "4-5": {"starts": 10, "top3": 1},  # 10% -> -5
            "6+": {"starts": 20, "top3": 1},  # 5% -> -8
        }
    }
    r1 = [r.items["R1"] for r in assess_risks(parse_race_card(data))]
    assert r1 == [0.0, -2.0, -5.0, -8.0]


def test_r1_is_unknown_without_data_and_zero_with_too_few_starts():
    runners = [_runner(1, "Prep", 1), _runner(2, "Other", 1), {"number": 3, "horse": "H3"}]
    data = small_card(runners)
    data["rotation_trends"] = {"Prep": {"1": {"starts": 2, "top3": 0}}}
    r1 = [r.items["R1"] for r in assess_risks(parse_race_card(data))]
    assert r1 == [0.0, None, None]


def test_r6_workout_and_weight():
    runners = [
        {"number": 1, "horse": "a", "workout_rating": "D", "horse_weight_change": -12},
        {"number": 2, "horse": "b", "workout_rating": "C"},
        {"number": 3, "horse": "c", "workout_rating": "A", "horse_weight_change": 4},
        {"number": 4, "horse": "d"},
    ]
    r6 = [r.items["R6"] for r in assess_risks(parse_race_card(small_card(runners)))]
    assert r6 == [-6.0, -2.0, 0.0, None]  # -5 - 3 capped at -6


def test_r8_class_record():
    def run(grade, finish):
        return {"date": "2026-04-01", "finish": finish, "grade": grade}

    runners = [
        {"number": 1, "horse": "a", "past_runs": [run("G1", 2)]},
        {"number": 2, "horse": "b", "past_runs": [run("G1", 7)]},
        {"number": 3, "horse": "c", "past_runs": [run("G2", 1)]},
        {"number": 4, "horse": "d", "past_runs": [run("G3", 5)]},
    ]
    r8 = [r.items["R8"] for r in assess_risks(parse_race_card(small_card(runners)))]
    assert r8 == [0.0, -3.0, -3.0, -5.0]


def test_verdict_and_sigma_cap():
    r = RunnerRisk(1, "x", {"R1": -8.0, "R2": -6.0, "R8": -5.0, "R3": None}, {})
    assert r.total == -19.0
    assert r.verdict[0] == "drop"
    assert r.sigma_points() == 5.0  # 3 items x 2, capped
    assert r.unknown == ["R3"]
    assert RunnerRisk(1, "x", {"R1": -2.0}, {}).verdict[0] == "low"


# ----------------------------------------------------------------- scenarios


def test_scenarios_sum_to_one_and_soft_going_is_e(card):
    sim = simulate_race(card, 20_000, seed=1)
    probs = [s["probability"] for s in sim.scenarios()]
    assert len(probs) == len(SCENARIOS)
    assert sum(probs) == pytest.approx(1.0)
    pace, bias = np.array([-1.0, 0.0, 1.0, -1.0]), np.array([1.0, 0.0, 0.0, -1.0])
    params = ShockParams()
    assert classify_scenarios("good", pace, bias, params).tolist() == [3, 1, 2, 0]
    assert classify_scenarios("heavy", pace, bias, params).tolist() == [4] * 4
    assert simulate_race(card, 1000, seed=1, shocks=False).scenarios() == []


# ----------------------------------------------------------------- uncertainty


def test_harville_top3_matches_trifecta():
    p = np.array([0.4, 0.25, 0.15, 0.12, 0.08])
    tri = harville_trifecta(p)
    expected = np.array(
        [sum(tri[o].sum() for o in [np.s_[i], np.s_[:, i], np.s_[:, :, i]]) for i in range(5)]
    )
    got = harville_top3(p)[0]
    np.testing.assert_allclose(got, expected, atol=1e-12)
    assert got.sum() == pytest.approx(3.0)


def test_zero_tau_reproduces_softmax(card):
    ev = evaluate_card(card)
    params = UncertaintyParams(0.0, 0.0, 0.0, 0.0, n_draws=50)
    u = score_uncertainty(ev, params=params)
    p = np.exp(ev.log_s) / np.exp(ev.log_s).sum()
    np.testing.assert_allclose(u.win_mean, p, atol=1e-12)
    np.testing.assert_allclose(u.win_lo, u.win_hi, atol=1e-12)


def test_intervals_widen_with_risk(card):
    ev = evaluate_card(card)
    risks = assess_risks(card, ev.raw)
    u = score_uncertainty(ev, risks, UncertaintyParams(n_draws=2000))
    assert np.all(u.win_lo <= u.win_mean) and np.all(u.win_mean <= u.win_hi)
    base = score_uncertainty(ev, None, UncertaintyParams(n_draws=2000))
    assert np.all(u.tau >= base.tau)


# ----------------------------------------------------------------- market


def test_odds_ladder_marks_gaps():
    ladder = odds_ladder([1, 2, 3, 4], ["a", "b", "c", "d"], [2.0, 2.4, 6.0, 6.5])
    assert [r["cluster"] for r in ladder] == [1, 1, 2, 2]
    assert [r["gap_before"] for r in ladder] == [False, False, True, False]
    assert ev_label(1.6) == "妙味大" and ev_label(0.5) == "割高"


def test_overbet_flag():
    view = market_view(
        [1, 2, 3, 4, 5],
        list("abcde"),
        [2.0, 4.0, 6.0, 8.0, 20.0],
        win=np.array([0.2, 0.3, 0.2, 0.2, 0.1]),
        win_lo=np.array([0.1, 0.2, 0.1, 0.1, 0.05]),
        fundamental_win=np.array([0.05, 0.3, 0.25, 0.25, 0.15]),
    )
    flags = {r["number"]: r["r9_overbet"] for r in view["runners"]}
    assert flags[1] and not any(flags[n] for n in (2, 3, 4, 5))
    assert market_view([1], ["a"], None, None, None, None) == {"available": False}


# ----------------------------------------------------------------- marks and tickets


def test_article_marks_respect_risk():
    win = np.array([0.35, 0.25, 0.15, 0.1, 0.08, 0.07])
    top3 = np.array([0.7, 0.6, 0.45, 0.35, 0.3, 0.25])
    risk = np.array([-13.0, -2.0, 0.0, -1.0, 0.0, -9.0])
    marks = article_marks(win, top3, win * 0.8, win * 1.2, risk)
    by_mark = {}
    for m, i in marks:
        by_mark.setdefault(m, []).append(i)
    assert by_mark["◎"] == [1]  # the best runner is excluded by its risk total
    assert 0 in by_mark["×"]
    assert 0 not in by_mark.get("○", []) + by_mark.get("▲", [])


def test_tickets_budget_and_returns():
    odds = np.array([3.0, 4.5, 6.0, 9.0, 15.0, 30.0])
    win = 0.78 / odds
    win = win / win.sum()
    marks = [("◎", 0), ("○", 1), ("▲", 2), ("△", 3), ("△", 4)]
    tickets = build_tickets(marks, odds)
    assert {t["bet"] for t in tickets} == {"win", "quinella", "trio"}
    assert all(t["stake"] % 100 == 0 and t["stake"] >= 100 for t in tickets)
    assert abs(sum(t["stake"] for t in tickets) - 10_000) <= 500
    assert sum(t["bet"] == "trio" for t in tickets) <= 1 + 6
    summary = evaluate_tickets(tickets, win)
    assert 0 <= summary["p_total_loss"] <= 1 and summary["expected_return"] > 0
    hit = payout(tickets, (0, 1, 3))
    assert hit["win"] > 0 and hit["quinella"] > 0 and hit["trio"] > 0
    scen = {s["code"]: s for s in return_scenarios(marks, tickets, win)}
    assert scen["C"]["returns"]["win"] == 0 and scen["A"]["total"] > 0
    # the ticket hit probabilities agree with an enumeration of Harville orders
    tri = harville_trifecta(win)
    trio_t = next(t for t in tickets if t["bet"] == "trio")
    assert trio_t["hit_probability"] == pytest.approx(
        sum(tri[o] for o in permutations(trio_t["runners"]))
    )


# ----------------------------------------------------------------- report


def test_analyze_and_render(card, tmp_path):
    a = analyze(card, n_sims=20_000, budget=5000)
    json.dumps(a, ensure_ascii=False, default=int)
    assert {m["mark"] for m in a["marks"]} >= {"◎", "○", "▲"}
    assert a["market"]["available"] and a["tickets"]
    assert len(a["runners"]) == card.field_size
    html = render_html(a, "# 見出し\n\n本文**強調**\n\n- 箇条")
    assert "印一覧" in html and "<strong>強調</strong>" in html and "<li>箇条</li>" in html


def test_cli(tmp_path, capsys):
    out = tmp_path / "analysis.json"
    main(["analyze", str(EXAMPLE), "--out", str(out), "--sims", "5000"])
    assert "Risk sheet" in capsys.readouterr().out
    main(["render", str(out), "--html", str(tmp_path / "r.html")])
    assert (tmp_path / "r.html").read_text("utf-8").startswith("<!doctype html>")


def test_markdown_table_and_code():
    html = markdown_to_html("|a|b|\n|-|-|\n|1|2|\n\n```\nx < y\n```")
    assert "<table>" in html and "<td>1</td>" in html and "x &lt; y" in html
