# Usage guide

How to run aikeiba: set it up, predict a graded race, check the system on a past race,
and fit the weights once history data exists. For how predictions are made and scored, see
[methodology.md](methodology.md); for the analysis steps, see
[race-analysis-workflow.md](race-analysis-workflow.md).

## What to expect today

- The pipeline runs end to end, from collecting a race card to a PDF report.
- **The probabilities are not validated yet.** No real history is connected, so the factor
  weights are hand-set priors. The output says so (`weights=prior`, `validated: false`). The
  value strategy then shows expected values but no stakes; read every probability as a
  reference value, not as a reason to bet.
- Race cards are collected by Claude Code agents reading public pages. Data that only paid
  pages carry (speed figures, graded-race records of jockeys and trainers, sire statistics by
  condition) is usually missing, and a missing factor counts as the field average.

## 1. One-time setup

### Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync
uv run pytest
```

### Network access (Claude Code on the web)

The default cloud environment cannot reach the racing sites. Use an environment whose
allowed hosts include the domains in `fetch.SITES`:

```
www.netkeiba.com  race.netkeiba.com  db.netkeiba.com  news.netkeiba.com
www.keibalab.jp   p.keibabook.co.jp  umanity.jp  uma-x.jp  note.com  www.jra.go.jp
```

A personal environment named `aikeiba` with exactly these hosts exists; start sessions for
this repository in it.

### Record the sites' terms as reviewed

`aikeiba-fetch` refuses every site whose terms you have not recorded as reviewed. Read each
site's terms of use, then record it yourself (or tell Claude explicitly which sites you have
reviewed):

```bash
uv run aikeiba-fetch --review-terms netkeiba --note "read on 2026-11-20"
uv run aikeiba-fetch --sites          # shows what is recorded
```

The record is kept in `data/scraping/terms_reviewed.json`. It is per environment and not
committed (`data/` is gitignored), so a new container needs it again.

### PDF reports

`aikeiba-report render --pdf` needs a Chromium binary. It is found automatically, or set
`AIKEIBA_CHROME` or pass `--chrome`. Without one, use the HTML report.

## 2. Predict an upcoming graded race

In Claude Code, ask for a race by name:

```
/predict-race マイルチャンピオンシップ
```

Optionally add a date or a budget (default 10,000 yen). The skill
(`.claude/skills/predict-race/SKILL.md`) runs these steps:

1. **Identify the race** and the data stage. Nominations come about a week ahead, the final
   field with gates two days before (for a Sunday G1, Friday 9:00 JST), odds once betting
   opens (for a Sunday G1, usually on Saturday), horse weights about an hour before post time.
   The cutoff is the time of the request; nothing published after it is used.
2. **Collect the card.**
   - The `race-card-collector` agent writes `base.json`: the field, the going, past-edition
     trends and the course trends.
   - Then one `runner-profile-collector` agent per four horses writes `runners-<n>.json`, in
     parallel.
   - Everything goes to `data/race_cards/<date>-<slug>/`.
3. **Merge and check** the card, then **simulate** (1,000,000 trials) and **analyse** (risk
   sheet, scenarios, marks, tickets).
4. **Write the article** and render the report as HTML/PDF.

A full collection takes 10–15 minutes of agent work. If a usage limit interrupts the agents,
ask Claude to continue: the agents resume from where they stopped, and fetched pages are
cached.

### Check the merged card before simulating

The collectors work independently, so check that every runner's fields use **one definition**
before running the simulation. Every factor is compared within the field, so mixed
definitions shift runners against each other:

- `stats` records: the same period for every runner (e.g. `jockey_year` = this year's JRA
  record). If only some runners have a record, or the periods differ, drop that record for
  all.
- `workout_rating`: one source and one mapping for the whole field (e.g. スポーツ報知's
  G/A/B/C mapped G→A, A→B, B→C, C→D).
- `past_runs.course`: one naming style (English, as in `base.json`).
- `past_runs.going`: full codes (良/稍重/重/不良); netkeiba's short 稍/不 are rejected.

### Run the steps by hand

```bash
D=data/race_cards/2026-11-22-mile-championship
uv run aikeiba-card merge $D/base.json $D/runners-*.json -o $D/card.json
uv run aikeiba-card validate $D/card.json
uv run aikeiba-simulate $D/card.json --sims 1000000 --json $D/prediction.json
uv run aikeiba-report analyze $D/card.json --budget 10000 --out $D/analysis.json
uv run aikeiba-report render $D/analysis.json --article $D/article.md \
  --html $D/report.html --pdf $D/report.pdf
```

| Command | Main options |
| --- | --- |
| `aikeiba-card merge BASE FRAGMENTS... -o OUT` | assembles and validates a card |
| `aikeiba-card validate CARD` | checks that a card parses |
| `aikeiba-simulate CARD` | `--sims` (default 1,000,000), `--seed`, `--json PATH`, `--history`, `--weights`, `--no-shocks` |
| `aikeiba-report analyze CARD --out PATH` | `--budget`, `--sims`, `--seed`, `--history`, `--weights` |
| `aikeiba-report render ANALYSIS --html PATH` | `--article`, `--pdf`, `--chrome` |
| `aikeiba-fetch URL` | `--format all/text/tables/html`, `--max-age` (hours), `--sites`, `--review-terms SITE` |

The card format is in `examples/race_card.example.json`. The loader rejects a card with a
past run on or after the race day, a workout after it, or a finish outside the field size;
fix the card, never the check.

### Read the output

- **`weights=prior` / `validated`**: whether the weights are hand-set or fitted, and whether
  they have beaten the market walk-forward. Only validated weights give stakes.
- **Win / top-2 / top-3 per runner**: `win_se` is the sampling error of the 1,000,000 trials.
  More trials do not make the prediction more accurate.
- **Marks**: three strategies.
  - 的中重視 (hit): the most likely to finish well.
  - 両立 (balanced): the same, skipping runners the market overrates.
  - 回収重視 (value): expected value; needs odds.
- **`reasons`**: the factors that moved a runner's score most.
- **`factors unknown for every runner`**: factors that contributed nothing.
- **Analysis**:
  - the risk sheet (R1–R8; `不明` = unknown, never guessed)
  - scenario probabilities A–E (D only on good going, E only on soft or heavy)
  - 90% intervals of the win probability
  - marks ◎○▲△☆×
  - tickets: quinella and trio odds are estimates from win odds, marked as such

### Race-day update

When odds, horse weights, the going, the cushion value or paddock reports come out, add them
to the card (re-run the collectors, or fill fields from a cited source), then run the
simulation and the analysis again. With odds on every runner, the market is combined into the
strengths and expected values appear.

## 3. Check the system on a past race

```
/verify-race 天皇賞（秋） 2025
```

The prediction is built with a cutoff on the evening before. The skill freezes it with
sha256, and only then fetches the result and scores it:

```bash
uv run aikeiba-evaluate $D/prediction.json $D/result.json --log data/predictions/log.csv
uv run aikeiba-evaluate --summary
```

The summary gives hit rates per strategy against the favourite, and calibration (ECE). Treat
this log as a check of the pipeline rather than as evidence of skill:

- pages show statistics as of today, which include the race itself;
- one race says almost nothing (±10 points on a top-3 rate even after 100 races).

## 4. Fit the weights from history

When a history table exists (one row per runner per past race, as parquet), fit the weights
and use them:

```bash
uv run aikeiba-train --history data/history.parquet --out data/models/weights.json
uv run aikeiba-simulate card.json --history data/history.parquet --weights data/models/weights.json
```

- **Required columns**: `race_id`, `race_date`, `race_name`, `course`, `surface`,
  `distance`, `going`, `horse_id`, `horse`, `finish_position`.
- **Optional columns**: listed in `history.OPTIONAL` (jockey, trainer, sire, odds, times,
  passing order and more). Each one adds factors.
- **How history is used**: `--history` fills the card from history as of the race date
  (`history.enrich_card`), the same way training does, so a card collected by the agents
  gains the full past runs and records.
- **The training report** compares the fitted weights with the prior weights, a uniform
  guess and the market, year by year. It then says whether fitted+market beats the market
  with confidence; only then are the weights `validated`. Without `--history` it runs on
  synthetic data, as a smoke test only.

Other commands: `uv run aikeiba-backtest` (LightGBM baseline against the market) and
`uv run aikeiba-train` without arguments both run on synthetic data.

## 5. Data sources in practice

From the first live run (Mile Championship 2026, collected on 2026-11-20):

| Source | Status |
| --- | --- |
| netkeiba | Main source: entries, past runs, pedigree, jockey/trainer pages, news (dated workout reports and post-work comments). Workout and comment pages are paid or open only two days before at 13:00; past-run tables need `/horse/result/<id>/`. |
| JRA | Going, cushion and moisture (from `/keiba/baba/_data_cushion.html` and `_data_moist.html`; the main page loads them by JavaScript), past-edition data. Some this-week pages return 403; archive PDFs cannot be read (see below). |
| umanity | Past-edition analysis and course data (running style, gate, sires) for the course trends. |
| keibabook | Blocked: robots.txt disallows all pages for other user agents. |
| keibalab | Unreachable from the cloud environment: the site resets the connection during TLS. |
| uma-x, note | Public predictions, but rarely dated before the cutoff, so `consensus_share` usually stays empty. |

## 6. Known issues

Found in the first live run; until fixed, work around them as described.

| Issue | Workaround |
| --- | --- |
| Tickets can exceed the budget (each ticket is rounded up to 100 yen; 10,200 yen for a 10,000-yen budget) | Trim the smallest tickets by hand |
| Before odds exist, return scenarios show 0 yen instead of "not computed" | Ignore them until odds are in the card |
| A return scenario's third place can be a runner marked × | Read the scenarios with the marks |
| Course names are compared as strings, so 京都 and Kyoto differ | Use English course names in `past_runs` |
| netkeiba's short going codes 稍 / 不 are rejected | Write 稍重 / 不良 |
| `aikeiba-fetch` prints a traceback on HTTP or network errors | Treat it as "page unavailable" |
| `aikeiba-fetch` cannot read PDFs and caches them corrupted (JRA cushion archive) | Past-run cushion values stay empty |
| Text extraction loses links, some netkeiba pedigree tables, and shifts the columns of netkeiba result tables | Read those pages with `--format html` |
| `closing_speed` compares last-3F times across distances and going without adjustment, which penalises runners from longer or softer races | Discount it when reading `reasons` |
| A missing margin (common for overseas runs) leaves `margin_form` computed from the wins only | Be careful with runners whose runs are mostly overseas |
| `workout_score` mixes a time z-score (when peers worked on the same course) with the A–E rating | Compare workout scores with care |
| The analysis report shows an expected return and yen allocation even with unvalidated weights | Read the tickets as an example allocation, not a recommendation |

## 7. Troubleshooting

| Message | Cause and fix |
| --- | --- |
| `refused: the terms of <site> have not been recorded as reviewed` | Record the review (section 1) |
| `refused: robots.txt of <host> disallows ...` | The site does not allow it; use another source |
| `ConnectionError` / 403 from the proxy | The host is not allowed in the environment, or the site refuses cloud access |
| `past run of <horse> on <date> is not before the race` | A collector read a page updated after the race; drop that run |
| `unknown going '稍'` | Use the full going code |
| Agents stop with a usage-limit error | Wait for the reset, then ask Claude to continue |
