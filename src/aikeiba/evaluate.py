"""Score a frozen prediction against the actual result, and keep a log.

The prediction is the JSON written by ``aikeiba-simulate --json``. The result is
written by the race-result-checker agent:

    {"race": "...", "date": "YYYY-MM-DD",
     "finish": [{"number": 16, "position": 1}, ...],
     "sources": ["https://..."]}

Scratched runners are simply absent from ``finish``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

_EPS = 1e-12


def evaluate(prediction: dict, result: dict) -> dict:
    position = {f["number"]: f["position"] for f in result["finish"]}
    if not position:
        raise ValueError("result has no finishers")
    predicted = {r["number"] for r in prediction["runners"]}
    unknown = set(position) - predicted
    if unknown:
        raise ValueError(f"result has horse numbers not in the prediction: {sorted(unknown)}")
    runners = [r for r in prediction["runners"] if r["number"] in position]  # drop scratches
    win_total = sum(r["win"] for r in runners)
    winners = [n for n, p in position.items() if p == 1]
    top3 = sorted(n for n, p in position.items() if p <= 3)
    order = sorted(position, key=position.get)

    def prob(r, key="win"):
        return r[key] / win_total if key == "win" else r[key]

    by_number = {r["number"]: r for r in runners}
    model_rank = {
        r["number"]: i + 1 for i, r in enumerate(sorted(runners, key=lambda r: -r["win"]))
    }
    marks = {p["mark"]: p["number"] for p in prediction["picks"]}
    trio_sets = [tuple(sorted(c["numbers"])) for c in prediction.get("trio", [])]
    trifecta = [tuple(c["numbers"]) for c in prediction.get("trifecta", [])]
    quinella = [tuple(sorted(c["numbers"])) for c in prediction.get("quinella", [])]

    out = {
        "race": prediction["race"],
        "date": prediction["date"],
        "weights": prediction.get("weights"),
        "n_runners": len(runners),
        "actual_top3": "-".join(map(str, order[:3])),
        "honmei_finish": position.get(marks.get("◎")),
        "taikou_finish": position.get(marks.get("○")),
        "tanana_finish": position.get(marks.get("▲")),
        "honmei_won": position.get(marks.get("◎")) == 1,
        "honmei_top3": (position.get(marks.get("◎")) or 99) <= 3,
        "marks_in_top3": sum((position.get(n) or 99) <= 3 for n in marks.values()),
        "quinella_hit": tuple(sorted(order[:2])) in quinella,
        "trio_hit_rank": (trio_sets.index(tuple(top3)) + 1) if tuple(top3) in trio_sets else None,
        "trifecta_hit_rank": (
            trifecta.index(tuple(order[:3])) + 1 if tuple(order[:3]) in trifecta else None
        ),
        "winner_model_rank": min(model_rank[w] for w in winners),
        "winner_prob": sum(prob(by_number[w]) for w in winners),
        "log_loss": -math.log(max(sum(prob(by_number[w]) for w in winners), _EPS)),
        "top3_mean_place_prob": sum(by_number[n]["top3"] for n in top3) / len(top3),
    }
    if all("market_win" in r for r in runners):
        m_total = sum(r["market_win"] for r in runners)
        m_prob = sum(by_number[w]["market_win"] for w in winners) / m_total
        market_rank = {
            r["number"]: i + 1
            for i, r in enumerate(sorted(runners, key=lambda r: -r["market_win"]))
        }
        out["winner_market_rank"] = min(market_rank[w] for w in winners)
        out["market_winner_prob"] = m_prob
        out["market_log_loss"] = -math.log(max(m_prob, _EPS))
    return out


def append_log(row: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            rows = [
                r for r in csv.DictReader(f) if (r["race"], r["date"]) != (row["race"], row["date"])
            ]
    rows.append({k: "" if v is None else v for k, v in row.items()})
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def summarize(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)

    def mean(key):
        vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
        return sum(vals) / len(vals) if vals else None

    return {
        "races": n,
        "honmei_win_rate": sum(r["honmei_won"] == "True" for r in rows) / n if n else None,
        "honmei_top3_rate": sum(r["honmei_top3"] == "True" for r in rows) / n if n else None,
        "trio_hit_rate": sum(r["trio_hit_rank"] != "" for r in rows) / n if n else None,
        "mean_log_loss": mean("log_loss"),
        "mean_market_log_loss": mean("market_log_loss"),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Score a prediction against the result")
    parser.add_argument("prediction", help="JSON from aikeiba-simulate --json")
    parser.add_argument("result", help="result JSON from the race-result-checker")
    parser.add_argument("--log", default="data/predictions/log.csv")
    args = parser.parse_args(argv)

    row = evaluate(
        json.loads(Path(args.prediction).read_text("utf-8")),
        json.loads(Path(args.result).read_text("utf-8")),
    )
    append_log(row, args.log)
    for key, value in row.items():
        print(f"{key:>22}: {value:.4f}" if isinstance(value, float) else f"{key:>22}: {value}")
    print("\nlog summary:", json.dumps(summarize(args.log), ensure_ascii=False))


if __name__ == "__main__":
    main()
