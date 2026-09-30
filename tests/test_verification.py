import json
from pathlib import Path

import pytest

from aikeiba.cardtools import merge
from aikeiba.evaluate import append_log, evaluate, summarize
from aikeiba.racecard import load_race_card
from aikeiba.simulate import simulate_race

EXAMPLE = Path(__file__).parent.parent / "examples" / "race_card.example.json"


def base_card():
    return {
        "race": {
            "name": "t",
            "grade": "G1",
            "date": "2026-09-27",
            "course": "Nakayama",
            "surface": "芝",
            "distance": 1200,
            "going": "良",
        },
        "runners": [{"number": i, "horse": f"H{i}", "draw": i} for i in (1, 2, 3, 4)],
        "sources": ["https://base"],
    }


def test_merge_fills_runner_fields_and_unions_sources():
    frag_a = {
        "runners": [{"number": 1, "horse": "H1", "jockey": "J1", "draw": 8}],
        "sources": ["https://a"],
    }
    frag_b = {"runners": [{"number": 3, "trainer": "T3"}], "sources": ["https://a", "https://b"]}
    card = merge(base_card(), [frag_a, frag_b])
    assert card["runners"][0]["jockey"] == "J1"
    assert card["runners"][0]["draw"] == 1  # base value wins
    assert card["runners"][2]["trainer"] == "T3"
    assert card["sources"] == ["https://base", "https://a", "https://b"]


def test_merge_rejects_mismatches():
    with pytest.raises(ValueError, match="not in the base card"):
        merge(base_card(), [{"runners": [{"number": 9}]}])
    with pytest.raises(ValueError, match="'X'"):
        merge(base_card(), [{"runners": [{"number": 1, "horse": "X"}]}])


@pytest.fixture(scope="module")
def prediction():
    result = simulate_race(load_race_card(EXAMPLE), 50_000, seed=0)
    return json.loads(json.dumps(result.to_dict(), default=int))


def test_evaluate_scores_picks_and_probabilities(prediction):
    honmei = prediction["picks"][0]["number"]
    others = [r["number"] for r in prediction["runners"] if r["number"] != honmei]
    finish = [{"number": honmei, "position": 1}] + [
        {"number": n, "position": i + 2} for i, n in enumerate(others)
    ]
    row = evaluate(prediction, {"race": "t", "date": "2026-10-04", "finish": finish})
    assert row["honmei_won"] and row["honmei_finish"] == 1
    assert row["winner_model_rank"] == 1
    assert 0 < row["winner_prob"] < 1 and row["log_loss"] > 0
    assert "market_log_loss" in row


def test_evaluate_handles_scratches(prediction):
    numbers = [r["number"] for r in prediction["runners"]]
    finish = [{"number": n, "position": i + 1} for i, n in enumerate(numbers[1:])]
    row = evaluate(prediction, {"race": "t", "date": "d", "finish": finish})
    assert row["n_runners"] == len(numbers) - 1
    with pytest.raises(ValueError, match="not in the prediction"):
        evaluate(prediction, {"race": "t", "date": "d", "finish": [{"number": 99, "position": 1}]})


def test_log_replaces_same_race_and_summarizes(tmp_path, prediction):
    log = tmp_path / "log.csv"
    finish = [
        {"number": r["number"], "position": i + 1} for i, r in enumerate(prediction["runners"])
    ]
    row = evaluate(prediction, {"race": "t", "date": "d", "finish": finish})
    append_log(row, log)
    append_log(row, log)  # same race again replaces the row
    summary = summarize(log)
    assert summary["races"] == 1
    assert summary["honmei_top3_rate"] == 1.0
