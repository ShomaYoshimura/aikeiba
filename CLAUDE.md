# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

aikeiba aims to build a system that predicts Japanese horse racing (競馬) results with AI, focusing on graded races (重賞). The repository is at the **design stage**: no Python implementation, package manifest, build tooling, linter, or tests exist yet. The design documents and comments are written in Japanese; keep new ones in Japanese too.

The only substantive file is `Keiba simulator architecture.jsx`. It is a self-contained React component: a single `export default function App()` that uses only `useState` from `react`, has no other dependencies, and does all styling inline through the `COLORS` token object. It is the architecture blueprint ("重賞予想シミュレーター 完全アーキテクチャ", v1.0) and renders as an interactive page. There is no bundler config. To view it, paste it into a React sandbox or artifact renderer, or scaffold a Vite/React app around it.

## The blueprint is the spec

The blueprint drives future implementation. Its content lives in data constants at the top of the file, and the JSX below only renders them:

- `layers`: 7 layers (L0–L6). Each has `modules`, and each module lists `name`, `tech`, `desc`, `inputs`, `outputs`, and an optional `detail`. A module's `inputs`/`outputs` artifact names (e.g. `raw_races.parquet`, `feature_store`, `elo_ratings.db`, `ranking_scores`, `calibrated_probs`) define the data contracts between modules.
- `dataFlows`: the layer-to-layer edges.
- `techStack`: the planned libraries.
- `kpis`: the target metrics.
- The validation workflow steps are written inline in the `kpi` view.

To change the design, edit these constants rather than the render code. If you rename an artifact, update every module that reads or writes it.

### Planned pipeline (L0 → L6)

1. **L0 データ収集・前処理**: gets data from JRA-VAN, netkeiba, and TARGET; computes a speed index; turns pedigrees into vectors with Word2Vec; stores features in a Feast/DuckDB feature store that guarantees no future leakage.
2. **L1 実力評価**: ELO ratings kept separately for each course × distance × going combination. K is dynamic: 64 for a G1 win, 16 for condition races. Also speed-index trend analysis.
3. **L2 適性評価**: a course-fit logistic regression, a KNN search for similar past races (K=15), a Random Forest pedigree-fit scorer, and a GPyTorch Gaussian-process imputer for horses with little data. The imputer also outputs an uncertainty σ.
4. **L3 予測モデル**: four models: LightGBM LambdaMART (optimizes NDCG@3), a lifelines CoxPH hazard model, a PyMC Bayesian posterior (2000 MCMC draws), and a Mesa agent-based simulator.
5. **L4 シミュレーション**: a dynamic pace generator, common shocks (track and inside/outside bias), and a Numba-JIT Monte Carlo run of 50,000 trials with t-distributed noise. A stacking ensemble combines the four L3 models with weights Ranking 35%, Bayes 25%, ABS 25%, Hazard 15%.
6. **L5 キャリブレーション・評価**: isotonic or Platt calibration, bootstrap confidence intervals, and drift monitoring with MLflow and Evidently.
7. **L6 出力**: a probability-matrix CSV, a pace-sensitivity heatmap, a model-consensus report (Jinja2), and a Streamlit/FastAPI dashboard.

Planned stack: Python 3.11, Pandas/Polars/DuckDB, LightGBM, scikit-learn, PyMC, GPyTorch, NumPy/Numba, Mesa, lifelines, MLflow, Evidently AI, Feast, DVC, FastAPI, Streamlit, Docker, GitHub Actions.

### Non-negotiable design constraints

- **No future leakage.** Features and simulations may use only data that was available at the time of the race. The feature store is designed to enforce this.
- **Strict time-based split.** Train on 2021–2023 G1/G2 races and validate on 2024–2025. Backtest all 7 methods on their own, plus the ensemble (8 patterns in total).
- **Calibration is part of the output.** A simulated 30% win probability must mean a 30% real win rate. Check the calibration curve at the 10/20/30/40% bands.
- **KPI targets:** Top-3 hit rate > 55%, Brier score < 0.12 (the odds-inverse baseline is 0.16), rank correlation > 0.45, and hit rate > 65% when all four models agree.
