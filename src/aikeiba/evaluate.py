"""Score a frozen prediction against the actual result, and keep logs.

The prediction is the JSON written by ``aikeiba-simulate --json``. The result is
written by the race-result-checker agent:

    {"race": "...", "date": "YYYY-MM-DD",
     "finish": [{"number": 16, "position": 1}, ...],
     "payouts": {"win": {"16": 1850}},      # optional, yen per 100 yen
     "sources": ["https://..."]}

Scratched runners are simply absent from ``finish``.

Two logs are kept: one row per race (hits per strategy, the favourite as the baseline,
log loss) and one row per runner (predicted probabilities and the outcome), from which
calibration over all logged races is computed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import pandas as pd

from aikeiba.metrics import calibration_table, expected_calibration_error
from aikeiba.strategies import STRATEGIES

_EPS = 1e-12


def _strategy_rows(prediction: dict) -> dict:
    strategies = prediction.get("strategies")
    if strategies is None:  # predictions written before strategies existed
        strategies = {"hit": prediction["picks"]}
    return strategies


def evaluate(prediction: dict, result: dict) -> dict:
    position = {f["number"]: f["position"] for f in result["finish"]}
    if not position:
        raise ValueError("result has no finishers")
    unknown = set(position) - {r["number"] for r in prediction["runners"]}
    if unknown:
        raise ValueError(f"result has horse numbers not in the prediction: {sorted(unknown)}")
    runners = [r for r in prediction["runners"] if r["number"] in position]  # drop scratches
    by_number = {r["number"]: r for r in runners}
    win_total = sum(r["win"] for r in runners)
    winners = [n for n, p in position.items() if p == 1]
    order = sorted(position, key=position.get)
    top3 = sorted(n for n, p in position.items() if p <= 3)
    model_rank = {
        r["number"]: i + 1 for i, r in enumerate(sorted(runners, key=lambda r: -r["win"]))
    }
    winner_prob = sum(by_number[w]["win"] for w in winners) / win_total

    out = {
        "race": prediction["race"],
        "date": prediction["date"],
        "weights": prediction.get("weights"),
        "n_runners": len(runners),
        "actual_top3": "-".join(map(str, order[:3])),
        "winner_model_rank": min(model_rank[w] for w in winners),
        "winner_prob": winner_prob,
        "log_loss": -math.log(max(winner_prob, _EPS)),
        "top3_mean_place_prob": sum(by_number[n]["top3"] for n in top3) / len(top3),
    }

    payouts = {int(k): v for k, v in (result.get("payouts") or {}).get("win", {}).items()}
    for name, picks in _strategy_rows(prediction).items():
        picks = [p for p in picks if p["number"] in position]
        marks = {p["mark"]: p["number"] for p in picks}
        honmei = position.get(marks.get("◎"))
        out[f"{name}_honmei_finish"] = honmei
        out[f"{name}_honmei_won"] = honmei == 1
        out[f"{name}_honmei_top3"] = honmei is not None and honmei <= 3
        out[f"{name}_marks_in_top3"] = sum(position[n] <= 3 for n in marks.values())
        if name == "value":
            returns = 0.0
            for p in picks:
                if position[p["number"]] == 1:
                    odds = by_number[p["number"]].get("win_odds")
                    returns += payouts.get(p["number"], (odds or 0) * 100) / 100
            out["value_bets"] = len(picks)
            out["value_return"] = returns
            out["value_payout_source"] = "result" if payouts else "card odds"

    trio = [tuple(sorted(c["numbers"])) for c in prediction.get("trio", [])]
    trifecta = [tuple(c["numbers"]) for c in prediction.get("trifecta", [])]
    quinella = [tuple(sorted(c["numbers"])) for c in prediction.get("quinella", [])]
    out["quinella_hit"] = tuple(sorted(order[:2])) in quinella
    out["trio_hit_rank"] = trio.index(tuple(top3)) + 1 if tuple(top3) in trio else None
    out["trifecta_hit_rank"] = (
        trifecta.index(tuple(order[:3])) + 1 if tuple(order[:3]) in trifecta else None
    )

    if all("market_win" in r for r in runners):
        m_total = sum(r["market_win"] for r in runners)
        m_prob = sum(by_number[w]["market_win"] for w in winners) / m_total
        favourite = max(runners, key=lambda r: r["market_win"])["number"]
        market_rank = {
            r["number"]: i + 1
            for i, r in enumerate(sorted(runners, key=lambda r: -r["market_win"]))
        }
        out["favourite_finish"] = position[favourite]
        out["favourite_won"] = position[favourite] == 1
        out["favourite_top3"] = position[favourite] <= 3
        out["winner_market_rank"] = min(market_rank[w] for w in winners)
        out["market_winner_prob"] = m_prob
        out["market_log_loss"] = -math.log(max(m_prob, _EPS))
    return out


def runner_rows(prediction: dict, result: dict) -> list[dict]:
    """One row per finisher: predicted probabilities (renormalized after scratches) and outcome."""
    position = {f["number"]: f["position"] for f in result["finish"]}
    runners = [r for r in prediction["runners"] if r["number"] in position]
    win_total = sum(r["win"] for r in runners)
    m_total = sum(r.get("market_win", 0.0) for r in runners)
    return [
        {
            "race": prediction["race"],
            "date": prediction["date"],
            "number": r["number"],
            "win": r["win"] / win_total,
            "top3": r["top3"],
            "market_win": r["market_win"] / m_total if m_total else "",
            "finish": position[r["number"]],
            "won": int(position[r["number"]] == 1),
            "placed": int(position[r["number"]] <= 3),
        }
        for r in runners
    ]


def _write_replacing(rows: list[dict], path: Path, race: str, date: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            old = [r for r in csv.DictReader(f) if (r["race"], r["date"]) != (race, date)]
    rows = old + [{k: "" if v is None else v for k, v in r.items()} for r in rows]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def append_log(row: dict, path: str | Path, runners: list[dict] | None = None) -> None:
    """Add a race to the race log (and its runners to the runner log next to it),
    replacing any earlier entry for the same race and date."""
    path = Path(path)
    _write_replacing([row], path, row["race"], row["date"])
    if runners is not None:
        _write_replacing(runners, runner_log_path(path), row["race"], row["date"])


def runner_log_path(race_log: str | Path) -> Path:
    race_log = Path(race_log)
    return race_log.with_name(race_log.stem + "_runners" + race_log.suffix)


def _rate(rows, key) -> float | None:
    vals = [r[key] == "True" for r in rows if r.get(key) not in (None, "")]
    return sum(vals) / len(vals) if vals else None


def summarize(path: str | Path) -> dict:
    """Hit rates per strategy vs. the favourite, trio hits, log loss, and calibration."""
    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)

    def mean(key):
        vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
        return sum(vals) / len(vals) if vals else None

    summary = {"races": n}
    for name in STRATEGIES:
        summary[f"{name}_honmei_win_rate"] = _rate(rows, f"{name}_honmei_won")
        summary[f"{name}_honmei_top3_rate"] = _rate(rows, f"{name}_honmei_top3")
    bets = sum(int(r["value_bets"]) for r in rows if r.get("value_bets"))
    returns = sum(float(r["value_return"]) for r in rows if r.get("value_return"))
    summary["value_bets"] = bets
    summary["value_roi"] = returns / bets if bets else None
    summary["favourite_win_rate"] = _rate(rows, "favourite_won")
    summary["favourite_top3_rate"] = _rate(rows, "favourite_top3")
    summary["trio_hit_rate"] = sum(r["trio_hit_rank"] != "" for r in rows) / n if n else None
    summary["mean_log_loss"] = mean("log_loss")
    summary["mean_market_log_loss"] = mean("market_log_loss")

    runner_log = runner_log_path(path)
    if runner_log.exists():
        r = pd.read_csv(runner_log)
        summary["win_ece"] = expected_calibration_error(r["win"], r["won"])
        summary["top3_ece"] = expected_calibration_error(
            r["top3"], r["placed"], bins=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
        )
        if r["market_win"].notna().all():
            summary["market_win_ece"] = expected_calibration_error(r["market_win"], r["won"])
    return summary


def calibration_report(path: str | Path) -> pd.DataFrame:
    """Predicted vs. observed win rate per bin over every logged runner."""
    r = pd.read_csv(runner_log_path(path))
    return calibration_table(r["win"], r["won"])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Score a prediction against the result")
    parser.add_argument("prediction", nargs="?", help="JSON from aikeiba-simulate --json")
    parser.add_argument("result", nargs="?", help="result JSON from the race-result-checker")
    parser.add_argument("--log", default="data/predictions/log.csv")
    parser.add_argument("--summary", action="store_true", help="only print the log summary")
    args = parser.parse_args(argv)

    if not args.summary:
        if not (args.prediction and args.result):
            parser.error("prediction and result are required unless --summary is given")
        prediction = json.loads(Path(args.prediction).read_text("utf-8"))
        result = json.loads(Path(args.result).read_text("utf-8"))
        row = evaluate(prediction, result)
        append_log(row, args.log, runner_rows(prediction, result))
        for key, value in row.items():
            text = f"{value:.4f}" if isinstance(value, float) else value
            print(f"{key:>24}: {text}")
        print()
    print("log summary:")
    for key, value in summarize(args.log).items():
        text = f"{value:.4f}" if isinstance(value, float) else value
        print(f"{key:>24}: {text}")
    print("\nwin calibration (all logged runners):")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(calibration_report(args.log).to_string())


if __name__ == "__main__":
    main()
