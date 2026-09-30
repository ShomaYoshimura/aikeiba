# aikeiba

AIを用いて競馬を予想することを目的としたシステム構築

An AI-driven system for predicting JRA graded races (重賞). It turns model scores into
calibrated win probabilities and judges every change against the betting market.

## Status

Early stage. The foundation and a baseline pipeline exist and run end to end on synthetic
data. Real JRA data is not connected yet. See [ROADMAP.md](ROADMAP.md) for what comes next and
[docs/architecture](docs/architecture) for the full design.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync                      # install dependencies
uv run pytest                # run the tests
uv run aikeiba-backtest      # walk-forward backtest on synthetic data
uv run aikeiba-backtest --data runners.parquet --test-years 2023 2024
```

### Predict a race from Claude Code

In Claude Code, run `/predict-race <race name>` (for example `/predict-race 天皇賞（秋）`), or just
ask for a prediction of a named race. The `predict-race` skill collects the entries, jockeys,
recent form and going from public sources into a race card, then runs:

```bash
uv run aikeiba-simulate data/race_cards/<card>.json --sims 1000000
```

The system targets graded races (G1/G2/G3). Each runner is abstracted into a vector of 29
factors: past performance and aptitude, pedigree, trainer, owner, breeder and region, jockey
form and head-to-head record, draw and race-edition trends, workouts, a scored stable comment
and the share of public predictions. The simulation runs 1,000,000 trials of the finishing
order (about 2 seconds) with shared pace and track-bias shocks, and picks ◎○▲ with the factors
behind each pick, plus the most likely quinella, trio and trifecta. See
`examples/race_card.example.json` for the card format.

### Test on a past race

`/verify-race <race name> <date>` builds the prediction from information published before the
race only, freezes it, then fetches the result and scores it with `aikeiba-evaluate`, which
appends to `data/predictions/log.csv` and prints a running summary. Data collection is split
across Claude Code agents in `.claude/agents/` (`race-card-collector`, parallel
`runner-profile-collector`s, and `race-result-checker`, which runs only after the prediction is
frozen), so results cannot leak into the inputs.

How predictions are made and scored (three marking strategies; hit rate and calibration as the
primary scores) is described in [docs/methodology.md](docs/methodology.md).

### Fit the factor weights

```bash
uv run aikeiba-train --history data/history.parquet --out data/models/weights.json
uv run aikeiba-simulate card.json --history data/history.parquet --weights data/models/weights.json
```

Every past graded race becomes a training example, with its factors computed as they were on
race day. The weights are fitted by Plackett-Luce likelihood of the first three places, and a
walk-forward report compares them with the prior weights, a uniform guess and the market.
Without `--history` it runs on synthetic data. Until weights are fitted on real data, the
simulation uses hand-set prior weights and says so.

The backtest prints, for each test year, win log loss, Brier score, top-pick place rate and
value-bet ROI for the model and for market-implied probabilities, over all races and over
graded races only.

### Architecture blueprint

```bash
cd docs/architecture
npm install
npm run dev
```

## How it works

| Module | Role |
| --- | --- |
| `schema.py` | Runner table contract: one row per horse per race |
| `leakage.py` | Fails if any value was not available before post time |
| `features.py` | Form features computed from each horse's earlier races only |
| `baseline.py` | LightGBM LambdaMART; softmax temperature fitted on held-out recent races |
| `probability.py` | Per-race normalization, market probabilities, Harville, Plackett-Luce sampling |
| `metrics.py` | Win log loss, Brier score, place rate, value-bet ROI, calibration table |
| `validation.py` | Walk-forward splits by year |
| `backtest.py` | Model vs. market report (`aikeiba-backtest`) |
| `racecard.py` | Race card JSON for one upcoming race |
| `history.py` | History table of past races and its derived columns |
| `conditions.py`, `stats.py` | Condition bins, similarity, shrunk rates, profile vectors, head-to-head |
| `factors.py` | The 29 factors per runner, from history or from the card |
| `model.py` | Within-race standardization and the linear Plackett-Luce model |
| `train.py` | Fits factor weights on past graded races (`aikeiba-train`) |
| `strength.py` | Card → factors → log-strengths, with the reasons per runner |
| `simulate.py` | 1,000,000-trial Monte Carlo with pace and track-bias shocks (`aikeiba-simulate`) |
| `cardtools.py` | Merges and validates race cards from the collector agents (`aikeiba-card`) |
| `evaluate.py` | Scores a frozen prediction against the result and keeps a log (`aikeiba-evaluate`) |
| `strategies.py` | Three marking strategies: hit (的中重視), balanced (両立), value (回収重視) |

## Design principles

- **No future leakage.** Every feature carries the time it became available, and tests enforce it.
- **Train on all races, report on graded races.** Graded races alone (about 60 a year) are too
  few to train on.
- **The market is the benchmark.** Random guessing is not a meaningful baseline; odds are.
- **Baseline first.** Methods from the blueprint are added one at a time and kept only if the
  walk-forward backtest improves.

## Data

Race data is licensed (for example, JRA-VAN Data Lab) and is never committed to this
repository. See [ingest/windows](ingest/windows) for the planned JV-Link ingestion worker.
