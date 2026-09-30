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
