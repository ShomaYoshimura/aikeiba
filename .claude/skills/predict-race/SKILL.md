---
name: predict-race
description: Predict a JRA graded race (G1/G2/G3) from its name. Collects entries, horse/jockey/trainer/pedigree data, workouts, stable comments and public prediction trends into a race card, abstracts them into factor vectors, runs a 1,000,000-trial Monte Carlo simulation, scores a risk sheet for every runner, and writes a prediction article with marks, budgeted tickets and expected returns (optionally as a PDF report). Use when the user names a race (e.g. "天皇賞（秋）を予想して", "/predict-race 有馬記念").
---

# Predict a graded race

Input: a race name, optionally with a date or year, and optionally a budget (default
10,000 yen). Reply to the user in Japanese.

The steps follow the analysis workflow in `docs/race-analysis-workflow.md` (steps 1–11):
collection (steps 1–2), the model and the risk sheet (2.3–6), odds and tickets (7–8), the
article (10) and the report (11). Do the steps in order; odds are looked at only from
step 7 on, and the fundamental evaluation never uses them.

The system targets graded races only. If the race is not G1/G2/G3, say so and ask before
continuing.

## 1. Identify the race

- Resolve the name to one running: the next upcoming one unless the user gives a date.
  Confirm grade, course, surface, distance and date.
- Entries are published in stages. Nominations (特別登録) come out about a week ahead; the
  final field with gates (枠順確定) usually comes out two days before; odds exist once betting
  opens; horse weights about an hour before post time. Tell the user which stage the data is
  at, and simulate whatever field is known.

## 2. Collect the race card

Delegate collection to the collector agents so that each works in its own context. Use
`data/race_cards/<date>-<slug>/` as the working directory (`data/` is gitignored), and set
the **cutoff** to now (or, for a past race, see the verify-race skill).

1. Run the `race-card-collector` agent: race, cutoff, output `<dir>/base.json`. It also
   collects the trends of past editions of the race (`trends`) and of reference races at the
   same course, surface and distance (`course_trends`, `course_draw_stats`), and on race day
   the earlier races of the day (`same_day_races`) for today's track bias.
2. Split the field into groups of about four horses and run one `runner-profile-collector`
   agent per group **in parallel**: race conditions, cutoff, the horses (number, name,
   jockey), output `<dir>/runners-<n>.json`.
3. Merge and validate:

```bash
uv run aikeiba-card merge <dir>/base.json <dir>/runners-*.json -o <dir>/card.json
```

If an agent reports missing fields, you may fill them yourself from a cited source, under
the same rules: nothing published after the cutoff, **never guess or invent a value**, read
pages with `uv run aikeiba-fetch` from the sources in `docs/data-sources.md`, and add every
URL to `sources`. If the agents report that a site was refused because its terms are not
recorded as reviewed, tell the user (`aikeiba-fetch --review-terms <site>`, see
`docs/data-sources.md`); never record it yourself. A missing value counts as the
field average. The field list and the comment-scoring rubric are in the agent definitions
(`.claude/agents/`); the card format is in `examples/race_card.example.json`.

## 3. Simulate and analyse

```bash
uv run aikeiba-simulate <dir>/card.json --sims 1000000 --json <dir>/prediction.json \
  [--history data/history.parquet] [--weights data/models/weights.json]
uv run aikeiba-report analyze <dir>/card.json --budget <yen> --out <dir>/analysis.json \
  [--history data/history.parquet] [--weights data/models/weights.json]
```

`analysis.json` adds the risk sheet (R1–R8 for every runner), the scenario probabilities
(A–E), the fundamental win probability without odds, the 90% score-uncertainty interval
and robustness order, the odds ladder with its gaps and expected values, the report marks
(◎○▲△☆×), the budgeted tickets with estimated odds, and returns A/B/C. Read the printed
sheets before writing anything.

Pass `--history` and `--weights` when those files exist: statistics then come from the full
history (the card's `stats` and `trends` are only a fallback), and the factor weights are the
ones fitted by `aikeiba-train`. If the card is rejected, fix the card (never the validation)
and run again.

## 4. Article and report (in Japanese)

Write the prediction article to `<dir>/article.md` following
[article.md](article.md) (template, sources in `analysis.json`, checklist), and show it to
the user. Then, unless the user does not want it, lay it out as a report:

```bash
uv run aikeiba-report render <dir>/analysis.json --article <dir>/article.md \
  --html <dir>/report.html --pdf <dir>/report.pdf
```

The PDF needs a Chromium binary (found automatically, or `AIKEIBA_CHROME`); if none is
found, give the HTML. After the article, add a short technical summary:

1. The race, the data stage, the going, and the sources.
2. The reference-race tendencies at this course and distance that matter for this field
   (running style, gate, sire), briefly, from `course_trends` / `course_draw_stats`.
3. Every runner's win, top-2 and top-3 probability.
4. The marks of all three strategies in `strategies` (see `docs/methodology.md`):
   的中重視 (hit), 両立 (balanced) and 回収重視 (value, with expected value and Kelly stake;
   it may be empty). Explain each ◎ with its `reasons` (the factors that moved its score
   most), translated into plain Japanese, plus any notable comment. Say where the strategies
   disagree and why (usually price).
5. The five most likely 馬連, 三連複 and 三連単 combinations.
6. Caveats, briefly:
   - Whether the weights are `prior` (hand-set, not validated) or `fitted` (and on how many races).
   - Factors that were unknown for every runner (`missing_factors`).
   - More trials only reduce sampling error; accuracy depends on the inputs.

## 5. Race-day update (optional, step 9)

When horse weights, the going, the cushion value, paddock reports or earlier races of the day
are announced, add them to the card (re-run the collectors or fill the fields from a cited
source), re-run step 3 and update the article. Keep the changes small and say what changed
and why (e.g. R4/R6 or the scenario mix moved).

After the race, the prediction can be scored with the verify-race skill (step 5 there), which
fetches the result and logs it with `aikeiba-evaluate`.
