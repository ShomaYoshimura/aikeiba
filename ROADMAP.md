# Roadmap

The architecture blueprint (`docs/architecture/`) describes the full system. This
roadmap is the order in which to build it. Two rules apply to every phase:

- **The market is the benchmark.** A change counts as an improvement only if it
  improves win log loss (or ROI) relative to market-implied probabilities in a
  walk-forward backtest.
- **Baseline first, then ablation.** Add one method at a time on top of the
  Phase 1 baseline and keep it only if the backtest improves.

## Phase 0: Foundation (done)

- [x] Python package with uv, ruff, pytest and GitHub Actions CI
- [x] Runner-table contract (`schema.py`) and point-in-time leakage checks (`leakage.py`)
- [x] Race-level metrics: win log loss, multi-class Brier, top-pick place rate, value-bet ROI,
      calibration table (`metrics.py`)
- [x] Probability layer: per-race softmax with fitted temperature, market-implied probabilities,
      Harville exacta/trifecta, Plackett-Luce sampling (`probability.py`)
- [x] Walk-forward splits by year (`validation.py`)
- [x] LightGBM LambdaMART baseline and a backtest CLI against the market, runnable on
      synthetic data (`baseline.py`, `backtest.py`, `synthetic.py`)
- [x] Blueprint v1.1: fixes to training data, KPIs, ensemble weights, probability model,
      data sources; viewable with Vite

## Race-day prediction (available now)

- [x] `/predict-race` Claude Code skill: race card from public sources → 1,000,000-trial
      Monte Carlo (`aikeiba-simulate`) → ◎○▲ and quinella/trio/trifecta probabilities
- [x] 55 factors per runner (horse incl. speed figures, margins, early position, race level and
      rotation, pedigree incl. siblings, connections, jockey incl. head-to-head and pairing,
      race trends, reference races at the same course and distance, today's track bias,
      paddock, odds movement, workouts, stable comments, public consensus), from history or from the card
- [x] `aikeiba-train`: Plackett-Luce fit of factor weights on past graded races, market
      combination, walk-forward report (verified on synthetic history)
- [ ] Fit the weights on real history (needs #2); then fit `ShockParams` the same way
- [x] Log each prediction and the result (`/verify-race`, `aikeiba-evaluate`), with collector
      and result agents kept apart so results cannot leak into predictions
- [x] `aikeiba-fetch`: polite fetching from netkeiba, keibalab, keibabook, umanity, uma-x, note
      and JRA (robots.txt, rate limit, cache, terms-review gate); sources in `docs/data-sources.md`
- [ ] Review each site's terms and allow its domains in the environment's network settings
- [ ] Run `/verify-race` on the recent graded races once network access to the sources is set
- [ ] Speed up factor building for large histories (currently about 0.4 s per race)

## Phase 1: Real-data MVP

Goal: the Phase 0 pipeline running on real JRA data, with an honest market comparison.

1. **Data source decision** (#1). Confirm the JRA-VAN Data Lab subscription and terms. Decide
   whether TARGET frontier CSV export is the interim source. Check terms of service before
   scraping any site.
2. **Runner-table loader** (#2). Convert the chosen source into the `schema.py` format for all JRA
   races from at least 2015, including win odds with the time they were observed.
3. **Data versioning** (#3). Keep data out of Git; track it with DVC and a private remote.
4. **Real-data backtest** (#4). Walk-forward over 2019–2025, trained on all races and reported for
   all races and the graded subset. Record runs in MLflow.

Exit criteria: a reproducible report comparing the baseline with the market on real data.
Whether it beats the market is not required yet; knowing the gap is.

## Phase 2: Core features and decision layer

Each item is merged only if it improves the Phase 1 backtest.

1. **JV-Link ingestion worker** (#5) on Windows (`ingest/windows/`), writing parquet with
   `available_at` timestamps.
2. **Speed index** (#6) with going (track condition) variants estimated from earlier races only.
   The figure is implemented (`history_metrics.speed_figures`); the ablation on real data is pending.
3. **Hierarchical ELO** (#7): a global rating plus condition offsets, with K tuned by walk-forward.
4. **Course-fit model** and **KNN similar-race features** (#8).
5. **Calibration** (#9) (isotonic/Platt) on out-of-fold predictions from all races, renormalized
   per race.
6. **Expected value and bet sizing** (#10): EV threshold and fractional Kelly chosen by backtest ROI
   and drawdown.
7. **Monitoring** (#11): market-relative log loss tracked per month.

Exit criteria: market-relative log loss improves on the Phase 1 baseline, and the calibration
curve stays within the bands in the blueprint.

## Phase 3: Advanced models (ablation-gated)

Build only after Phase 2, and keep each only if the ablation shows a gain (tracked in #12):

- PyMC Bayesian ability posterior
- Monte Carlo engine with common shocks and per-horse uncertainty (must match the
  Plackett-Luce closed form when shocks are zero)
- Stacking ensemble (regularized logistic regression on out-of-fold predictions)
- Gaussian-process imputation for horses with little data
- Pedigree embeddings and pedigree-fit scorer
- CoxPH hazard model (first check that lap data is granular enough)
- Mesa agent-based simulator
- Pace sensitivity map, consensus report, Streamlit/FastAPI dashboard

## Decisions for the repository owner

- License (#13; none is set, so the code is all-rights-reserved by default)
- Repository visibility, given the data licensing terms
- Budget for the JRA-VAN subscription and a Windows machine for ingestion
