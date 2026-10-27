"""Cross-check a race card before simulating it.

Runners of the same field often met in the same past races (trials, the Derby, ...).
Those shared races let the card check itself:

- race-level facts (field size, course, surface, distance, going, grade) must agree
  between runners; where some runners lack a value that the others agree on, it is filled;
- two runners cannot share a finishing position (except dead heats, which are reported);
- a finish cannot exceed the field size, a winner's margin must be 0, a runner's early
  position cannot exceed the field size;
- every past run must be dated before the race, and runs must be most recent first.

``audit`` never invents a value: it only copies a value that other runners' records of the
same race agree on, and it reports what it changed.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field

RACE_FIELDS = ("field_size", "course", "surface", "distance", "going", "grade")
COVERAGE_FIELDS = (
    "field_size", "course", "surface", "distance", "going", "jockey", "margin", "time",
    "early_position", "last3f", "popularity",
)  # fmt: skip
_NAME_ALIASES = {"日本ダービー": "東京優駿"}
_GENERIC = re.compile(r"新馬|未勝利|勝クラス|万下|^(\d歳)?(以上)?オープン$|^\d歳(以上)?$")


def race_key(run: dict) -> tuple[str, str]:
    """(date, normalized name): NFKC, no spaces, ステークス -> S, カップ -> C, common aliases."""
    name = "".join(unicodedata.normalize("NFKC", run.get("race_name") or "").split())
    for alias, canonical in _NAME_ALIASES.items():
        name = name.replace(alias, canonical)
    name = name.replace("ステークス", "S").replace("カップ", "C")
    if _GENERIC.search(name):
        # many races share a generic name on the same day: also require course and distance
        if not (run.get("course") and run.get("distance")):
            return run["date"], f"{name}#{id(run)}"
        name = f"{name}@{run['course']}{run['distance']}"
    return run["date"], name


def _group_shared(entries: list[tuple[tuple[str, str], str, dict]]) -> dict:
    """Group runs by race: same date and one normalized name a prefix of the other
    (弥生賞 / 弥生賞ディープインパクト記念)."""
    groups: dict[tuple[str, str], list] = {}
    for (date, name), who, run in sorted(entries, key=lambda e: len(e[0][1])):
        key = next(
            (
                k
                for k in groups
                if k[0] == date and (name.startswith(k[1]) or k[1].startswith(name))
            ),
            (date, name),
        )
        groups.setdefault(key, []).append((who, run))
    return groups


@dataclass
class AuditReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    filled: list[str] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def audit(card: dict, fix: bool = False) -> AuditReport:
    """Check ``card`` (the JSON dict); with ``fix``, fill agreed values and sort runs in place."""
    report = AuditReport()
    race_date = card["race"]["date"]
    named = []  # (race key, runner, run)

    for runner in card["runners"]:
        who = f"{runner['number']} {runner['horse']}"
        runs = runner.get("past_runs") or []
        dates = [r["date"] for r in runs]
        if dates != sorted(dates, reverse=True):
            report.warnings.append(f"{who}: past runs not most recent first")
            if fix:
                runs.sort(key=lambda r: r["date"], reverse=True)
                report.filled.append(f"{who}: sorted past runs")
        if len(set(dates)) != len(dates):
            report.errors.append(f"{who}: two past runs on the same date")
        for run in runs:
            where = f"{who} {run['date']} {run.get('race_name', '')}"
            if run["date"] >= race_date:
                report.errors.append(f"{where}: dated on or after the race (leak)")
            size = run.get("field_size")
            if size and run["finish"] > size:
                report.errors.append(f"{where}: finish {run['finish']} > field size {size}")
            if size and run.get("early_position") and run["early_position"] > size:
                report.errors.append(f"{where}: early position > field size")
            margin = run.get("margin")
            if run["finish"] == 1 and margin:
                report.errors.append(f"{where}: winner with margin {margin}")
            if margin is not None and margin < 0:
                report.errors.append(f"{where}: negative margin")
            if run.get("race_name"):
                named.append((race_key(run), who, run))

    for (date, name), entries in _group_shared(named).items():
        if len(entries) < 2:
            continue
        finishes = defaultdict(list)
        for who, run in entries:
            finishes[run["finish"]].append(who)
        for pos, whos in finishes.items():
            if len(whos) > 1:
                report.warnings.append(
                    f"{date} {name}: {', '.join(whos)} all finished {pos} (dead heat or error)"
                )
        for f in RACE_FIELDS:
            values = {run[f] for _, run in entries if run.get(f) is not None}
            if len(values) > 1:
                report.errors.append(
                    f"{date} {name}: runners disagree on {f}: {sorted(map(str, values))}"
                )
            elif len(values) == 1 and fix:
                value = values.pop()
                for who, run in entries:
                    if run.get(f) is None:
                        run[f] = value
                        report.filled.append(
                            f"{who} {date} {name}: {f} = {value} (from other runners)"
                        )
        sizes = {run.get("field_size") for _, run in entries} - {None}
        if sizes:
            size = max(sizes)
            if len(entries) > size:
                report.errors.append(
                    f"{date} {name}: {len(entries)} runners recorded, field of {size}"
                )

    all_runs = [r for runner in card["runners"] for r in runner.get("past_runs") or []]
    report.coverage = {
        "runners_with_runs": sum(bool(r.get("past_runs")) for r in card["runners"]),
        "runners": len(card["runners"]),
        "runs": len(all_runs),
        "runs_with_source": sum(bool(r.get("source")) for r in all_runs),
        **{
            f: round(sum(r.get(f) is not None for r in all_runs) / len(all_runs), 2)
            for f in COVERAGE_FIELDS
            if all_runs
        },
    }
    return report


def format_report(report: AuditReport) -> str:
    lines = [f"{'OK' if report.ok else 'ERRORS FOUND'}"]
    for title, items in (("errors", report.errors), ("warnings", report.warnings),
                         ("filled", report.filled)):  # fmt: skip
        if items:
            lines.append(f"\n{title} ({len(items)}):")
            lines += [f"  - {item}" for item in items]
    lines.append("\ncoverage: " + json.dumps(report.coverage, ensure_ascii=False))
    return "\n".join(lines)
