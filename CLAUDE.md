# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

aikeiba predicts JRA graded races (重賞). A Python package (`src/aikeiba`) runs a baseline pipeline end to end on synthetic data; real data is not connected yet. `ROADMAP.md` sets the build order, and `docs/architecture/keiba-simulator-architecture.jsx` is the full design (UI text in Japanese). Write code, comments and new docs in English, and reply to the user in Japanese.

## Commands

```bash
uv sync                                   # install (Python 3.11)
uv run pytest                             # all tests
uv run pytest tests/test_metrics.py::test_dead_heat_averages_over_winners   # one test
uv run ruff check . && uv run ruff format --check .
uv run aikeiba-backtest                   # walk-forward backtest on synthetic data
uv run aikeiba-backtest --data runners.parquet --test-years 2023 2024
uv run aikeiba-simulate examples/race_card.example.json --sims 1000000   # one race
uv run aikeiba-train --history data/history.parquet --out data/models/weights.json  # omit --history for synthetic

cd docs/architecture && npm install && npm run build   # blueprint viewer (Vite)
```

CI (`.github/workflows/ci.yml`) runs ruff, pytest, a synthetic backtest smoke test and the viewer build.

## Architecture

Every stage passes a **runner table**: a pandas DataFrame with one row per horse per race, whose columns are defined in `schema.py` (`race_id`, `post_time`, `horse_id`, `finish_position`, `win_odds`, `grade`, ...). New modules take and return this shape. Array-based functions take `race_ids` alongside the values and group by them.

Pipeline (`backtest.run_backtest`): `validate_runners` → `features.add_prior_form_features` → `validation.walk_forward_by_year` → `baseline.LambdaRankBaseline` fit per fold → `predict_proba` (per-race softmax) → `metrics` for both the model and `probability.market_probs`, reported for all races and for the graded subset.

Invariants that the code and tests rely on:

- **Point in time.** Every feature has an `available_at` column (e.g. `form_available_at`), and `leakage.assert_point_in_time` rejects values at or after `post_time`. Per-horse aggregates use strictly earlier races (cumulative sum minus the current row, or `shift(1)`). Tuning, temperature fitting, calibration and stacking use only the training fold. `LambdaRankBaseline.fit` fits its temperature on the latest 20% of training races with a model that has not seen them.
- **Probabilities sum to 1 within each race.** Scores become probabilities only through `softmax_by_race` or `normalize_by_race`. Market probabilities are normalized odds inverses, which removes the takeout.
- **One finishing-order model.** Plackett-Luce, with Harville as its closed form. `sample_plackett_luce` is the base for the Monte Carlo engine and must match Harville when no shocks are added.
- **The market is the benchmark.** Report metrics next to `market`, never against a random baseline. Train on all races; graded races (about 60 a year) are too few to train on alone.

**Race-day path (graded races only)**: the `/predict-race` skill (`.claude/skills/predict-race/SKILL.md`) collects a race card JSON (`racecard.py`) from public sources into `data/race_cards/` → `strength.evaluate_card` → `simulate.simulate_race`.

- `factors.compute_factors` abstracts the card, and the history table when given, into one raw value per factor per runner. `factors.FACTORS` is the single list of factors (horse form and aptitude, pedigree, connections, jockey including head-to-head, draw and race-edition trends, workouts, stable-comment score, public consensus), each with a prior weight. To add a factor, append it there and compute it in both the history path (`history_stat_factors` / `horse_factors`) and the card-only path (`card_stat_factors` / card fields).
- With history (`history.prepare_history`), statistics use only rows dated before the race (a prefix of the date-sorted table) and override the card's `stats`/`trends`. Rates are shrunk toward a prior (`stats.shrunk_rate`); sire/damsire use entity x condition-bin profile vectors (`stats.profile_matrix`, bins in `conditions.py`).
- `model.standardize` z-scores factors within the race (unknown = 0 = field average, clipped at ±3). `model.log_strengths` = x·w, or alpha·x·w + beta·log(market probability) when every runner has odds.
- `train.py` (`aikeiba-train`) fits w by Plackett-Luce likelihood of the top 3 on past graded races (examples) while statistics come from all races, fits alpha/beta on later races than w, reports walk-forward winner log loss against prior weights, uniform and the market, and saves `ModelWeights` JSON. Without a weights file the prior weights are used, and the output says so.
- Each simulation trial adds Gumbel noise (Plackett-Luce) and shared shocks: pace (number of front runners × running style), inside/outside track bias (wider on softer going) and form uncertainty shrinking with known runs. With `shocks=False` win rates must equal the softmax of the strengths; keep that test passing. `ShockParams` values are still unfitted priors.

`synthetic.py` exists so the pipeline and tests run without licensed data; it has no deliberate model edge. Real race data is never committed (`data/`, `*.parquet` are gitignored). JRA-VAN comes through JV-Link, a Windows-only COM component, so ingestion is planned as a separate Windows worker (`ingest/windows/`) that writes parquet.

## Blueprint

The blueprint's content lives in the data constants at the top of the JSX file (`PHASES`, `layers`, `dataFlows`, `techStack`, `kpis`); the JSX below only renders them, so edit the constants. Each module's `inputs`/`outputs` names are the data contracts between modules, so a rename must update every reader. `phase` (P1/P2/P3) matches the phases in `ROADMAP.md`; P3 modules are adopted only if an ablation shows a gain over the baseline. Keep the blueprint and the roadmap consistent when either changes.
