"""Race analysis report: every number the written article needs, in one ``analysis.json``.

``aikeiba-report analyze`` runs the race-day pipeline for one card and adds the parts of
the analysis workflow (``docs/race-analysis-workflow.md``) that the simulation alone does
not give:

- the risk sheet for every runner (``risk.py``, step 2.9),
- scenario probabilities and per-scenario rates (``simulate.py``, steps 4-5),
- the fundamental win probability without odds, the score-uncertainty intervals and
  the robustness order (``uncertainty.py``, step 6),
- the odds ladder with its gaps, expected values and R9 overbets (``market.py``, step 7),
- the report marks ◎○▲△☆× (``strategies.article_marks``, step 11-B),
- budgeted tickets with estimated odds and returns (``tickets.py``, step 8).

The article itself (step 10) is written by Claude from this file, following
``.claude/skills/predict-race/article.md``. ``aikeiba-report render`` lays the analysis
and the article out as HTML, and as PDF when a Chromium binary is available (step 11).
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from aikeiba.market import format_ladder, market_view
from aikeiba.model import ModelWeights
from aikeiba.racecard import load_race_card
from aikeiba.risk import RiskParams, assess_risks
from aikeiba.simulate import SimulationResult, simulate_race, style_scores
from aikeiba.strategies import article_marks
from aikeiba.tickets import (
    BET_LABELS,
    TicketParams,
    build_tickets,
    evaluate_tickets,
    return_scenarios,
)
from aikeiba.uncertainty import UncertaintyParams, score_uncertainty

CAVEATS = (
    "Harville法（Plackett-Luce）は上位人気馬の2・3着確率を過大評価しやすく、"
    "複勝・3連系の実配当はモデルより低く出やすい。",
    "馬連・3連複のオッズは単勝オッズからの推定値（Harville×払戻率）で、実オッズではない。",
    "リスク点数・不確実性τ・シナリオ区分の閾値は手設定の事前値で、まだ実データで当てはめていない。",
    "スコアは統計的傾向に基づく推定であり、個体差・調教・当日の状態で覆りうる。",
)


def _fundamental_win(sim: SimulationResult) -> np.ndarray:
    ev = sim.evaluation
    f = ev.x @ ev.weights.vector()
    e = np.exp(f - f.max())
    return e / e.sum()


def analyze(
    card,
    *,
    n_sims: int = 1_000_000,
    seed: int = 0,
    history: pd.DataFrame | None = None,
    weights: ModelWeights | None = None,
    budget: int = 10_000,
    risk_params: RiskParams | None = None,
    uncertainty_params: UncertaintyParams | None = None,
) -> dict:
    sim = simulate_race(card, n_sims, seed, history=history, weights=weights)
    card = sim.card  # filled from history when given, so the risk sheet sees the same runs
    ev = sim.evaluation
    style = style_scores(card, ev.raw["early_speed"].to_numpy())
    risks = assess_risks(card, ev.raw, style, risk_params)
    unc = score_uncertainty(ev, risks, uncertainty_params, seed=seed)
    # The epistemic spread scaled onto the simulated win rate (which includes race shocks).
    ratio_lo = unc.win_lo / np.maximum(unc.win_mean, 1e-12)
    ratio_hi = unc.win_hi / np.maximum(unc.win_mean, 1e-12)
    win_lo, win_hi = sim.win * ratio_lo, sim.win * ratio_hi
    fundamental = _fundamental_win(sim)
    odds = sim.odds
    numbers = [r.number for r in card.runners]
    horses = [r.horse for r in card.runners]
    risk_total = np.array([r.total for r in risks])

    marks = article_marks(sim.win, sim.top3, win_lo, win_hi, risk_total, odds)
    mark_of = {}
    for m, i in marks:
        mark_of.setdefault(i, m)
    market = market_view(numbers, horses, odds, sim.win, win_lo, fundamental)
    honmei = next((i for m, i in marks if m == "◎"), None)
    ev_honmei = None if odds is None or honmei is None else float(sim.win[honmei] * odds[honmei])
    params = TicketParams(budget=budget)
    tickets = build_tickets(marks, odds, ev_honmei, params)
    ticket_summary = evaluate_tickets(tickets, sim.win)
    scenarios = return_scenarios(marks, tickets, sim.win)

    def num(i):
        return numbers[i]

    runners = []
    for i, r in enumerate(card.runners):
        runners.append(
            {
                "number": r.number,
                "horse": r.horse,
                "jockey": r.jockey,
                "draw": r.draw,
                "running_style": r.running_style,
                "sire": r.sire,
                "damsire": r.damsire,
                "odds": r.win_odds,
                "mark": mark_of.get(i),
                "fundamental_win": float(fundamental[i]),
                "win": float(sim.win[i]),
                "win_lo": float(win_lo[i]),
                "win_hi": float(win_hi[i]),
                "top2": float(sim.top2[i]),
                "top3": float(sim.top3[i]),
                "top3_lo": float(unc.top3_lo[i]),
                "top3_hi": float(unc.top3_hi[i]),
                "tau": float(unc.tau[i]),
                "ev": None if odds is None else float(sim.win[i] * odds[i]),
                "ev_lo": None if odds is None else float(win_lo[i] * odds[i]),
                "risk": risks[i].to_dict(),
                "reasons": ev.reasons(i, 5),
            }
        )

    return {
        "race": {
            "name": card.name,
            "grade": card.grade,
            "date": card.date,
            "post_time": card.post_time,
            "course": card.course,
            "surface": card.surface,
            "distance": card.distance,
            "going": card.going,
            "cushion": card.cushion,
            "moisture": card.moisture,
            "weather": card.weather,
            "field_size": card.field_size,
        },
        "model": {
            "n_sims": n_sims,
            "seed": seed,
            "weights": "fitted" if ev.weights.fitted else "prior",
            "uses_market": ev.log_market is not None,
            "missing_factors": ev.missing_factors(),
            "uncertainty_draws": (uncertainty_params or UncertaintyParams()).n_draws,
        },
        "past_editions": list(card.past_editions),
        "scenarios": sim.scenarios(),
        "runners": runners,
        "robustness": [num(int(i)) for i in unc.robustness_order()],
        "market": market,
        "marks": [{"mark": m, "number": num(i), "horse": horses[i]} for m, i in marks],
        "tickets": [
            {
                "bet": t["bet"],
                "label": BET_LABELS[t["bet"]],
                "numbers": [num(i) for i in t["runners"]],
                "stake": t["stake"],
                "odds": t["odds"],
                "odds_estimated": t["bet"] != "win",
                "hit_probability": t.get("hit_probability"),
            }
            for t in tickets
        ],
        "ticket_summary": ticket_summary,
        "return_scenarios": [{**s, "order": [num(i) for i in s["order"]]} for s in scenarios],
        "combinations": {
            kind: [
                {"numbers": list(c), "probability": p}
                for c, p in sim.top_combinations(kind, 15 if kind == "trifecta" else 10)
            ]
            for kind in ("quinella", "trio", "trifecta")
        },
        "caveats": list(CAVEATS),
        "sources": list(card.sources),
    }


# ---------------------------------------------------------------- text output


def format_analysis(a: dict) -> str:
    race = a["race"]
    lines = [
        f"{race['name']} ({race['grade']})  {race['date']}  {race['course']} {race['surface']}"
        f" {race['distance']}m  going={race['going']}  {race['field_size']} runners",
        f"weights={a['model']['weights']}  market={'yes' if a['model']['uses_market'] else 'no'}",
        "",
        "Risk sheet (step 2.9; None = unknown):",
    ]
    risk_rows = [
        {
            "no": r["number"],
            "horse": r["horse"],
            **{k: "不明" if v is None else f"{v:.0f}" for k, v in r["risk"]["items"].items()},
            "total": r["risk"]["total"],
            "sigma": r["risk"]["sigma_points"],
            "R5": ",".join(r["risk"]["r5_tags"]),
            "verdict": r["risk"]["verdict_label"],
        }
        for r in a["runners"]
    ]
    lines.append(pd.DataFrame(risk_rows).to_string(index=False))
    lines += ["", "Scenarios (steps 4-5):"]
    for s in a["scenarios"]:
        lines.append(f"  {s['code']} {s['label']:<8} {s['probability']:.1%}")
    lines += ["", "Probabilities (step 6; interval = 90% score uncertainty):"]
    prob = pd.DataFrame(
        [
            {
                "mark": r["mark"] or "",
                "no": r["number"],
                "horse": r["horse"],
                "fund": r["fundamental_win"],
                "win": r["win"],
                "lo": r["win_lo"],
                "hi": r["win_hi"],
                "top3": r["top3"],
                "odds": r["odds"],
                "ev": r["ev"],
                "ev_lo": r["ev_lo"],
            }
            for r in a["runners"]
        ]
    ).sort_values("win", ascending=False)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        lines.append(prob.to_string(index=False))
    lines.append(f"\nRobustness (by lower win bound): {a['robustness'][:8]}")
    if a["market"].get("available"):
        marks = {r["number"]: r["mark"] for r in a["runners"] if r["mark"]}
        lines += ["", "Odds ladder (step 7):", format_ladder(a["market"]["ladder"], marks)]
        over = [r for r in a["market"]["runners"] if r["r9_overbet"]]
        if over:
            lines.append("R9 overbet: " + ", ".join(f"{r['number']} {r['horse']}" for r in over))
    lines += ["", "Marks: " + "  ".join(f"{m['mark']}{m['number']}" for m in a["marks"])]
    lines += ["", "Tickets (step 8; quinella/trio odds are estimates):"]
    for t in a["tickets"]:
        o = "" if t["odds"] is None else f" x{t['odds']:.1f}"
        hp = "" if t["hit_probability"] is None else f"  hit {t['hit_probability']:.1%}"
        lines.append(
            f"  {t['label']:<4} {'-'.join(map(str, t['numbers'])):<10} {t['stake']:>6}円{o}{hp}"
        )
    s = a["ticket_summary"]
    lines.append(f"  total {s['stake']}円")
    if s.get("estimated"):
        lines.append(
            f"  expected return x{s['expected_multiple']:.2f}  total loss {s['p_total_loss']:.0%}"
            f"  profit {s['p_profit']:.0%}"
        )
    for sc in a["return_scenarios"]:
        lines.append(f"  {sc['code']} {sc['label']} {sc['order']}: {sc['total']:.0f}円")
    return "\n".join(lines)


# ---------------------------------------------------------------- HTML / PDF


def markdown_to_html(text: str) -> str:
    """A small Markdown subset: headings, lists, tables, code blocks, bold, paragraphs."""
    out, para, in_code, in_list, table = [], [], False, False, []

    def inline(s):
        s = html.escape(s)
        return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)

    def flush():
        nonlocal para, in_list, table
        if para:
            out.append("<p>" + "<br>".join(inline(p) for p in para) + "</p>")
            para = []
        if in_list:
            out.append("</ul>")
            in_list = False
        if table:
            rows = [r for r in table if not re.fullmatch(r"\|?[\s:\-|]+\|?", r)]
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
            body = "".join(
                "<tr>"
                + "".join(
                    f"<{'th' if k == 0 else 'td'}>{inline(c)}</{'th' if k == 0 else 'td'}>"
                    for c in row
                )
                + "</tr>"
                for k, row in enumerate(cells)
            )
            out.append(f"<table>{body}</table>")
            table = []

    for line in text.splitlines():
        if line.startswith("```"):
            flush()
            out.append("</pre>" if in_code else "<pre>")
            in_code = not in_code
            continue
        if in_code:
            out.append(html.escape(line))
            continue
        if not line.strip():
            flush()
        elif m := re.match(r"(#{1,4})\s+(.*)", line):
            flush()
            out.append(f"<h{len(m[1]) + 1}>{inline(m[2])}</h{len(m[1]) + 1}>")
        elif line.lstrip().startswith(("- ", "* ")):
            if para or table:
                flush()
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(line.lstrip()[2:])}</li>")
        elif line.lstrip().startswith("|"):
            if para or in_list:
                flush()
            table.append(line.strip())
        else:
            para.append(line)
    flush()
    if in_code:
        out.append("</pre>")
    return "\n".join(out)


def _pct(x):
    return "" if x is None else f"{x:.1%}"


def _num(x, f="{:.2f}"):
    return "" if x is None else f.format(x)


_CSS = """
body { font-family: "Noto Sans CJK JP", "Hiragino Sans", "Yu Gothic", sans-serif;
  max-width: 900px; margin: 2em auto; padding: 0 16px; color: #1a1a1a; line-height: 1.7; }
h1 { border-bottom: 3px solid #b8860b; }
h2 { border-left: 6px solid #b8860b; padding-left: 8px; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.9em; }
th, td { border: 1px solid #ccc; padding: 4px 6px; text-align: left; }
th { background: #f3eee0; }
td.mark { font-weight: bold; font-size: 1.2em; }
pre { background: #f6f6f6; padding: 12px; overflow-x: auto; }
.cover { page-break-after: always; text-align: center; padding-top: 25vh; }
@media print { section { page-break-inside: avoid; } }
"""


def _table(header: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _risk_text(r: dict) -> str:
    risk = r["risk"]
    hits = [v for k, v in risk["notes"].items() if (risk["items"][k] or 0) < 0]
    return html.escape("／".join(hits))


def render_html(a: dict, article: str | None = None) -> str:
    race = a["race"]
    model = a["model"]
    title = f"{race['name']} {race['date'][:4]}"
    marks = _table(
        [
            "印",
            "馬番",
            "馬名",
            "勝率",
            "90%区間",
            "複勝率",
            "オッズ",
            "期待値",
            "リスク計",
            "主要リスク",
        ],
        [
            [
                r["mark"] or "",
                r["number"],
                html.escape(r["horse"]),
                _pct(r["win"]),
                f"{_pct(r['win_lo'])}–{_pct(r['win_hi'])}",
                _pct(r["top3"]),
                _num(r["odds"], "{:.1f}"),
                _num(r["ev"]),
                f"{r['risk']['total']:.0f}",
                _risk_text(r),
            ]
            for r in sorted(a["runners"], key=lambda r: -r["win"])
        ],
    ).replace("<tr><td>", "<tr><td class=mark>")
    tickets = _table(
        ["券種", "組み合わせ", "金額", "想定オッズ", "的中率"],
        [
            [
                t["label"],
                "-".join(map(str, t["numbers"])),
                f"{t['stake']:,}円",
                _num(t["odds"], "{:.1f}")
                + ("（推定）" if t["odds_estimated"] and t["odds"] else ""),
                _pct(t["hit_probability"]),
            ]
            for t in a["tickets"]
        ],
    )
    s = a["ticket_summary"]
    if s.get("estimated"):
        summary = (
            f"合計 {s['stake']:,}円 ／ 期待回収率 {s['expected_multiple']:.0%}"
            f" ／ 全滅確率 {s['p_total_loss']:.0%}"
        )
    else:
        summary = f"合計 {s['stake']:,}円（オッズ未確定のため回収率は未計算）"
    scen = _table(
        ["", "シナリオ", "確率"],
        [[x["code"], x["label"], _pct(x["probability"])] for x in a["scenarios"]],
    )
    robust = "、".join(str(n) for n in a["robustness"][:8])
    caveats = "".join(f"<li>{html.escape(c)}</li>" for c in a["caveats"])
    unknown = [r["horse"] for r in a["runners"] if len(r["risk"]["unknown"]) >= 4]
    if unknown:
        caveats += f"<li>リスク項目の多くが不明の馬：{html.escape('、'.join(unknown))}</li>"
    if model["weights"] == "prior":
        caveats += "<li>因子の重みは手設定の事前値（未検証）。</li>"
    article_html = (
        f"<section class=article>{markdown_to_html(article)}</section>" if article else ""
    )
    course = f"{race['course']} {race['surface']} {race['distance']}m"
    market = "あり" if model["uses_market"] else "なし"
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{_CSS}</style></head><body>
<div class=cover><h1>{html.escape(title)}</h1>
<p>{course} ／ 馬場 {race["going"]} ／ {race["field_size"]}頭</p>
<p>開催日 {race["date"]}　発走 {race["post_time"] or "不明"}</p></div>
{article_html}
<section><h2>印一覧</h2>{marks}</section>
<section><h2>買い目</h2>{tickets}<p>{summary}</p></section>
<section><h2>裏付け</h2><h3>展開シナリオ</h3>{scen}
<p>ロバスト性順（勝率区間の下限順）：{robust}</p>
<p>重み：{model["weights"]} ／ 市場併用：{market} ／ 試行 {model["n_sims"]:,}回
 ／ 不確実性 {model["uncertainty_draws"]:,}回</p></section>
<section><h2>留保・不確実性</h2><ul>{caveats}</ul></section>
</body></html>"""


def find_chrome() -> str | None:
    for name in ("chromium", "chromium-browser", "google-chrome", "chrome"):
        if path := shutil.which(name):
            return path
    env = os.environ.get("AIKEIBA_CHROME")
    if env:
        return env
    found = sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux*/chrome"))
    return found[-1] if found else None


def html_to_pdf(html_path: Path, pdf_path: Path, chrome: str | None = None) -> None:
    chrome = chrome or find_chrome()
    if chrome is None:
        raise RuntimeError("no Chromium found; set AIKEIBA_CHROME or pass --chrome")
    subprocess.run(
        [
            chrome,
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path.resolve()}",
            html_path.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Race analysis report (analysis.json, HTML, PDF)")
    sub = parser.add_subparsers(dest="command", required=True)
    an = sub.add_parser("analyze", help="run the analysis for a race card")
    an.add_argument("card")
    an.add_argument("--out", required=True, help="analysis JSON to write")
    an.add_argument("--budget", type=int, default=10_000)
    an.add_argument("--sims", type=int, default=1_000_000)
    an.add_argument("--seed", type=int, default=0)
    an.add_argument("--history")
    an.add_argument("--weights")
    rd = sub.add_parser("render", help="lay out an analysis (and article) as HTML / PDF")
    rd.add_argument("analysis")
    rd.add_argument("--article", help="the article in Markdown (step 10)")
    rd.add_argument("--html", required=True)
    rd.add_argument("--pdf")
    rd.add_argument("--chrome", help="Chromium binary for the PDF")
    args = parser.parse_args(argv)

    if args.command == "analyze":
        a = analyze(
            load_race_card(args.card),
            n_sims=args.sims,
            seed=args.seed,
            history=pd.read_parquet(args.history) if args.history else None,
            weights=ModelWeights.load(args.weights) if args.weights else None,
            budget=args.budget,
        )
        Path(args.out).write_text(
            json.dumps(a, ensure_ascii=False, indent=1, default=int), encoding="utf-8"
        )
        print(format_analysis(a))
        print(f"\nwrote {args.out}")
    else:
        a = json.loads(Path(args.analysis).read_text("utf-8"))
        article = Path(args.article).read_text("utf-8") if args.article else None
        out = Path(args.html)
        out.write_text(render_html(a, article), encoding="utf-8")
        print(f"wrote {out}")
        if args.pdf:
            html_to_pdf(out, Path(args.pdf), args.chrome)
            print(f"wrote {args.pdf}")


if __name__ == "__main__":
    main()
