import json
from pathlib import Path

import numpy as np
import pytest

from aikeiba.racecard import load_race_card, parse_race_card
from aikeiba.simulate import main, simulate_race
from aikeiba.strength import form_score, log_strengths, market_probabilities, record_score

EXAMPLE = Path(__file__).parent.parent / "examples" / "race_card.example.json"


@pytest.fixture
def card():
    return load_race_card(EXAMPLE)


def minimal_card(**runner_fields):
    runners = [
        {"number": i + 1, "horse": f"H{i + 1}", **{k: v[i] for k, v in runner_fields.items()}}
        for i in range(len(next(iter(runner_fields.values()))))
    ]
    race = {"name": "t", "date": "2026-01-01", "course": "c", "surface": "turf"}
    return parse_race_card({"race": {**race, "distance": 1600, "going": "良"}, "runners": runners})


def test_example_card_parses_japanese_aliases(card):
    assert card.going == "yielding"
    assert card.runners[0].running_style == "front"
    assert card.runners[3].running_style == "closer"


def test_rejects_invalid_cards():
    with pytest.raises(ValueError, match="going"):
        parse_race_card({"race": {"going": "mud"}, "runners": []})
    with pytest.raises(ValueError, match="three runners"):
        minimal_card(win_odds=[2.0, 3.0])


def test_without_shocks_win_rates_match_softmax_of_strengths(card):
    result = simulate_race(card, 200_000, seed=0, shocks=False)
    p = np.exp(log_strengths(card))
    np.testing.assert_allclose(result.win, p / p.sum(), atol=0.004)


def test_market_only_card_uses_market_strengths():
    card = minimal_card(win_odds=[2.0, 4.0, 8.0])
    p = np.exp(log_strengths(card))
    np.testing.assert_allclose(p / p.sum(), market_probabilities(card))


def test_without_odds_form_decides_strength():
    card = minimal_card(recent_finishes=[[1, 1, 2], [5, 6, 4], [9, 8, 10]])
    s = log_strengths(card)
    assert s[0] > s[1] > s[2]


def test_probabilities_are_consistent(card):
    r = simulate_race(card, 50_000, seed=1)
    assert r.win.sum() == pytest.approx(1.0)
    assert r.top2.sum() == pytest.approx(2.0)
    assert r.top3.sum() == pytest.approx(3.0)
    assert np.all(r.win <= r.top2) and np.all(r.top2 <= r.top3)
    assert r.trio.sum() == r.trifecta.sum() == r.quinella.sum() == 50_000


def test_same_seed_is_reproducible_across_chunk_boundaries(card):
    a = simulate_race(card, 30_000, seed=7, chunk_size=30_000)
    b = simulate_race(card, 30_000, seed=7, chunk_size=30_000)
    np.testing.assert_array_equal(a.trifecta, b.trifecta)
    c = simulate_race(card, 30_000, seed=7, chunk_size=7_000)  # different chunking, same model
    np.testing.assert_allclose(a.win, c.win, atol=0.01)


def test_fast_pace_favours_closers():
    styles = ["逃げ", "逃げ", "逃げ", "逃げ", "追込"]  # four front runners -> fast pace expected
    card = minimal_card(win_odds=[5.0] * 5, running_style=styles)
    r = simulate_race(card, 100_000, seed=0)
    assert r.win[4] > 0.2 + 0.01  # above the 20% it would get without the pace shock


def test_picks_are_distinct_and_follow_probabilities(card):
    r = simulate_race(card, 50_000, seed=2)
    picks = r.picks()
    assert len(set(picks)) == 3
    assert picks[0] == int(np.argmax(r.win))


def test_top_combinations_use_horse_numbers(card):
    r = simulate_race(card, 50_000, seed=3)
    (combo, p), *_ = r.top_combinations("trio")
    numbers = {x.number for x in card.runners}
    assert set(combo) <= numbers and len(set(combo)) == 3
    assert combo == tuple(sorted(combo))
    assert 0 < p < 1


def test_helpers():
    assert np.isnan(form_score(()))
    assert form_score((1,)) == 0.0
    assert np.isnan(record_score(None))


def test_cli_writes_json(tmp_path, capsys):
    out = tmp_path / "result.json"
    main([str(EXAMPLE), "--sims", "20000", "--json", str(out)])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [p["mark"] for p in data["picks"]] == ["◎", "○", "▲"]
    assert data["n_sims"] == 20000
    assert "◎" in capsys.readouterr().out
