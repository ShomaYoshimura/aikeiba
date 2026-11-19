# Race analysis workflow (v5.0)

The race-day analysis follows the "競馬分析専門AIプロンプト v5.0" workflow: steps 1–11, from
collecting information to a written prediction article and a PDF report. This page maps
each step to the code, and records where the workflow was adapted to the system's rules
(point in time, no invented values, fitted weights, the market as benchmark).

```
STEP 1   collect            race-card-collector / runner-profile-collector → card.json
STEP 2   race definition    card (course, going, trends, course_trends)      → article
STEP 2.3 last 3 editions    card.past_editions, card.trends                  → article
STEP 2.5 no-grade horses    risk R8 + uncertainty.no_grade_tau
STEP 2.7 pedigree           factors sire/damsire_aptitude, sibling_rate ...; risk R2/R4
STEP 2.8 jockey             factors jockey_* (never a reason to drop on its own)
STEP 2.9 risk sheet         risk.py (R1–R8, every runner)
STEP 3   scoring            factors.py → model.py (fitted or prior weights), no odds
STEP 4–5 scenarios          simulate.py pace / track-bias shocks, labelled A–E
STEP 6   probabilities      simulate.py (Plackett-Luce MC) + uncertainty.py (score doubt)
STEP 7   odds               market.py (ladder and gaps, EV, R9 overbet)
STEP 8   tickets            tickets.py (budget, estimated odds, returns A/B/C)
STEP 9   race-day update    card fields (horse_weight, paddock_score, same_day_races) → re-run
STEP 10  article            Claude, from analysis.json (.claude/skills/predict-race/article.md)
STEP 11  PDF                aikeiba-report render (HTML; PDF through Chromium)
```

Commands:

```bash
uv run aikeiba-report analyze <dir>/card.json --budget 10000 --out <dir>/analysis.json
uv run aikeiba-report render <dir>/analysis.json --article <dir>/article.md \
  --html <dir>/report.html --pdf <dir>/report.pdf
```

## Step by step

**STEP 1 — collection.** The collector agents fill the race card from public sources
before a cutoff (`docs/data-sources.md`). Unknown values are left out, never guessed; a
missing value counts as the field average in the model and as "不明" in the risk sheet.
The card now also carries the facts of the last editions (`past_editions`: winner,
popularity, running style, sire line, previous race, first/second half, pace, going, top-3
popularity and gates) and the past-edition record by previous race and previous finish
(`rotation_trends`, for R1).

**STEP 2 / 2.3 — race definition and the last three editions.** The structured parts are
the factors `race_trend` (10 editions: gate, age, sex, running style, last finish),
`course_trend` and `draw_bias` (reference races at the same course and distance). The
three-year review table and the "winner profile" are written in the article from
`past_editions`.

*Adapted:* the workflow's hand-set calibration coefficients (pace, style, gate, pedigree
and rotation tilts of up to ±5 points) are **not applied**. The same evidence already
enters the model through `race_trend` / `course_trend` / `draw_bias`, with shrinkage
toward a prior and fitted weights; adding tilts on top would count it twice. Whether the
model "worked" in past editions is measured by running `/verify-race` on them, not by a
judgement in the article.

**STEP 2.5 — horses new to the grade.** A runner that has never run at the race's grade
or above gets R8 (−3 or −5) and a wider score uncertainty (`no_grade_tau`). The
"trial-race lap comparison" and "prep-race level" of the workflow are covered by the
factors `speed_index`, `race_level` and `class_form`.

**STEP 2.7 / 2.8 — pedigree and jockey.** These are factors in `factors.FACTORS` (sire and
damsire progeny rate under today's condition bin, siblings, dam, inbreeding; jockey year,
graded, course, head-to-head and pairing rates). Their weights are fitted on past graded
races, not set by points. As the workflow requires, neither is ever a reason to drop a
runner on its own.

**STEP 2.9 — risk sheet (`risk.py`).** R1–R8 are scored for **every** runner from the card;
each value is a number (0 = checked, no risk) or unknown, so the sheet itself is the record
that the items were applied symmetrically. The total gives a verdict (−12 or less 消し濃厚,
−8 割引, −4 軽度警戒, otherwise リスク小) and R1/R2/R8 add to the score uncertainty
(σ +2 each, at most +5).

*Adapted:* the total is **not added to the strength**. Every item overlaps a factor the
model already weighs, so subtracting it as well would double-count. The risk total is used
for the marks (◎ needs ≥ −7; × at ≤ −12, or ≤ −8 for a fancied runner with expected value
below 1), for the bet axes and for the uncertainty. R5 (style vs. pace) is only a tag: the
pace shock of the simulation applies it, once, in every trial. R9 (overbet) is a flag in
step 7 and changes no score.

**STEP 3 — scoring without odds.** The fundamental score is x·w over the standardized
factors. `analysis.json` gives `fundamental_win` (softmax of the fundamental score, no
odds) next to the final probabilities, so the "aptitude rank vs. popularity" comparison of
step 7 uses a probability that never saw the odds.

*Adapted:* the workflow's 100-point hand scoring and the temperature T fitted to "top
runner wins 28–35%" are replaced by the factor model: weights fitted by Plackett-Luce
likelihood on past graded races (`aikeiba-train`), and the market combined only with a
fitted weight (`docs/methodology.md`).

**STEP 4–5 — scenarios.** The simulation's pace and track-bias shocks are continuous; each
trial is labelled A (slow), B (middle), C (fast), D (slow pace with an inside bias on good
going) or E (soft or heavy going), giving scenario probabilities and every runner's win
and top-3 rate per scenario. The labels add no effects of their own.

**STEP 6 — probabilities.** Luck of the race: 1,000,000 Plackett-Luce trials with shared
shocks (`simulate.py`). Doubt about the score: `uncertainty.py` draws each runner's
fundamental score 10,000 times with a runner-specific τ (less evidence, a new grade or
R1/R2/R8 flags widen it) and computes win (softmax) and top-3 (Harville) in closed form
per draw. The 5–95% spread, scaled onto the simulated win rate, is the interval in the
report; the robustness order ranks runners by its lower end. Harville overrates the
second and third places of favourites, so place and trio payouts tend to be tighter than
the model suggests; the report says so.

**STEP 7 — odds (`market.py`).** Odds are first used here. The ladder sorts runners by
odds and marks a gap where the next odds are at least 1.4 times the previous; expected
value is reported at the mean and at the lower end of the interval ("value only on
average" is flagged); R9 flags a runner the market ranks at least three places above its
fundamental rank with expected value below 1.

**STEP 8 — tickets (`tickets.py`).** About 30% of the budget on ◎ to win (40% when its
expected value is 2 or more), 30% on quinellas ◎-○, ◎-▲ and ○-▲ (the cover for ◎
missing), 40% on trios ◎○▲ and ◎-○/◎-▲ with each △ (at most six points). Stakes are
dutched within a bet type and rounded to 100 yen. Quinella and trio odds are estimates
(JRA payout rate / market Harville probability). Every first-three outcome is enumerated
under Harville on the model's win rates for the expected return, the chance of losing
everything and of a profit, and for three named outcomes A (◎-○-△), B (◎-▲-△) and
C (○-▲-△).

**STEP 9 — race-day update.** Add the announced horse weights, paddock scores, going,
cushion value and earlier races of the day (`same_day_races`) to the card and re-run;
R4/R6 and the scenario mix update with it. State what changed.

**STEP 10 — article.** Claude writes the article from `analysis.json` and the card,
following `.claude/skills/predict-race/article.md`: conclusion first, marks right after the
lead, the odds ladder drawn in the ◎ section, a separate section for fancied runners left
out (at least two independent reasons), the ticket table with stakes and estimated odds,
returns A/B/C, and an honest section on the outcomes that lose everything.

*Adapted:* the article is assertive, but every number in it comes from `analysis.json`, the
card or a cited source. A poor run is explained only by what the card records (trouble,
going, distance, class, pace); where nothing explains it, the article says the cause is
unknown rather than inventing one.

**STEP 11 — report.** `aikeiba-report render` lays out the cover, the article, the mark
table (win rate with interval, top-3, odds, expected value, risk total and risk reasons),
the tickets, the scenarios and robustness order, and the caveats as HTML, and prints it to
PDF with headless Chromium when one is found (`AIKEIBA_CHROME` or `--chrome` to point at
it).

## Marks (step 11-B, `strategies.article_marks`)

| Mark | Rule |
| --- | --- |
| ◎ | among the three most likely winners with risk total ≥ −7, the first whose interval (relative to its win rate) is no wider than the field's median |
| ○ | highest top-3 rate of the rest with expected value ≥ 1.0 (any, without odds) |
| ▲ | highest win rate of the rest with expected value ≥ 1.5 and popularity 4th or lower; else as ○ |
| △ | the next highest top-3 rates (at least 15%), up to four |
| ☆ | popularity 8th or lower, risk ≥ −3, rated at least four places above its popularity |
| × | risk ≤ −12, or risk ≤ −8 for a top-4 favourite with expected value below 1 |

These marks sit next to the three strategies of `docs/methodology.md` (hit / balanced /
value); the thresholds, the risk points and τ are initial choices to be tuned on the
prediction log.
