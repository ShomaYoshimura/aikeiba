"""Assemble a race card from the pieces written by the collector agents.

The race-card-collector writes a base card (race, field, trends); each
runner-profile-collector writes a fragment ``{"runners": [...]}`` for the horses it
was given. ``merge`` fills each base runner with the fragment fields for the same
horse number, keeps the base value where both have one, unions the sources, and
validates the result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aikeiba.audit import audit, format_report
from aikeiba.pastruns import past_runs_from_html
from aikeiba.racecard import parse_race_card


def merge(base: dict, fragments: list[dict]) -> dict:
    runners = {r["number"]: dict(r) for r in base["runners"]}
    sources = list(base.get("sources") or [])
    for frag in fragments:
        for r in frag.get("runners", []):
            if r["number"] not in runners:
                raise ValueError(f"fragment has horse number {r['number']} not in the base card")
            target = runners[r["number"]]
            if r.get("horse") and target.get("horse") and r["horse"] != target["horse"]:
                raise ValueError(
                    f"horse number {r['number']}: {r['horse']!r} vs {target['horse']!r}"
                )
            for key, value in r.items():
                if value is not None and target.get(key) is None:
                    target[key] = value
        sources += [s for s in frag.get("sources") or [] if s not in sources]
    card = {**base, "runners": [runners[n] for n in sorted(runners)], "sources": sources}
    parse_race_card(card)  # raises on anything invalid
    return card


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Race card tools")
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("merge", help="merge runner fragments into a base card")
    m.add_argument("base")
    m.add_argument("fragments", nargs="*")
    m.add_argument("-o", "--out", required=True)
    v = sub.add_parser("validate", help="check that a card parses")
    v.add_argument("card")
    a = sub.add_parser("audit", help="cross-check runners' past runs; --fix fills agreed values")
    a.add_argument("card")
    a.add_argument("--fix", action="store_true")
    a.add_argument("-o", "--out", help="where to write the fixed card (default: in place)")
    r = sub.add_parser("runs-from-html", help="past runs from a saved past-performance page")
    r.add_argument("html")
    r.add_argument("--number", type=int, required=True)
    r.add_argument("--horse", required=True)
    r.add_argument("--source", help="URL of the page, stored with each run")
    r.add_argument("--before", help="keep only runs before this date (YYYY-MM-DD)")
    r.add_argument("-o", "--out", required=True, help="runner fragment JSON to write")
    args = parser.parse_args(argv)

    if args.command == "merge":
        base = json.loads(Path(args.base).read_text("utf-8"))
        frags = [json.loads(Path(f).read_text("utf-8")) for f in args.fragments]
        card = merge(base, frags)
        Path(args.out).write_text(json.dumps(card, ensure_ascii=False, indent=1), "utf-8")
        filled = sum(bool(r.get("past_runs")) for r in card["runners"])
        print(f"wrote {args.out}: {len(card['runners'])} runners, {filled} with past runs")
    elif args.command == "validate":
        card = parse_race_card(json.loads(Path(args.card).read_text("utf-8")))
        print(f"ok: {card.name}, {card.field_size} runners")
    elif args.command == "audit":
        card = json.loads(Path(args.card).read_text("utf-8"))
        report = audit(card, fix=args.fix)
        print(format_report(report))
        if args.fix:
            parse_race_card(card)
            out = Path(args.out or args.card)
            out.write_text(json.dumps(card, ensure_ascii=False, indent=1), "utf-8")
            print(f"\nwrote {out}")
        if not report.ok:
            raise SystemExit(1)
    else:
        html = Path(args.html).read_text("utf-8", errors="replace")
        runs = past_runs_from_html(html, args.source)
        if args.before:
            runs = [run for run in runs if run["date"] < args.before]
        fragment = {
            "runners": [{"number": args.number, "horse": args.horse, "past_runs": runs[:10]}],
            "sources": [args.source] if args.source else [],
        }
        Path(args.out).write_text(json.dumps(fragment, ensure_ascii=False, indent=1), "utf-8")
        print(f"wrote {args.out}: {len(runs[:10])} runs")


if __name__ == "__main__":
    main()
