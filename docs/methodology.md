# Methodology: how predictions are made and scored

This page records the decisions behind the prediction logic and the evaluation. Change it
when a decision changes.

## How a prediction is made

1. **Abstract.** Everything known before the race becomes 29 factors per runner
   (`src/aikeiba/factors.py`). Statistics come from all races; rates with few observations are
   shrunk toward a prior so that a small sample cannot produce an extreme value.
2. **Compare within the race.** Each factor is z-scored within the field. A factor says how
   much better than *this field* a runner is; unknown values count as the field average.
3. **Score.** A linear model turns the factor vector into a log-strength (`model.py`). Linear
   was chosen because it can be fitted on a few hundred graded races and every pick can be
   explained by its factors. Non-linear models (e.g. gradient boosting) are to be compared
   once the history has thousands of races; interactions that matter can be added as explicit
   factors before that.
4. **Combine with the market (two stages).** The fundamental score is fitted first, without
   odds. When odds exist, the final log-strength is `alpha × fundamental + beta × log(market
   probability)`, with alpha and beta fitted on races later than those used for the
   fundamental weights (Benter's approach). The market is thus an input with a learned
   weight, not something the model copies.
5. **Simulate.** 1,000,000 Plackett-Luce trials with shared pace and track-bias shocks give
   win, top-2 and top-3 probabilities and bet-type frequencies (`simulate.py`). More trials
   only reduce sampling noise; accuracy comes from steps 1–4.
6. **Mark.** Three strategies turn the probabilities into ◎○▲ (`strategies.py`):

| Strategy | Purpose | ◎ | ○ / ▲ |
| --- | --- | --- | --- |
| 的中重視 (hit) | finish well, regardless of price | highest win probability | highest top-2 / top-3 probability of the rest |
| 両立 (balanced) | like hit, but avoid runners the market clearly overrates | as hit, among runners with expected value ≥ 0.9 | as hit, same filter |
| 回収重視 (value) | long-run return | highest expected value (probability × odds) above 1.1 | next highest; fractional Kelly (¼) stake |

The thresholds are initial choices, to be tuned on the prediction log.

### Qualitative information

- Stable comments are scored −2..+2 by a fixed rubric (in the `runner-profile-collector`
  agent), judging only what the comment says. The text is kept so the score can be audited.
- Public predictions become `consensus_share`, with a small prior weight: they mostly repeat
  the odds. Both weights are to be fitted once enough annotated races exist.

## How predictions are scored

The primary scores are **hit rate** and **calibration**. Log loss is reported alongside
because it is the objective the weights are fitted on.

| Score | What it measures | Baseline |
| --- | --- | --- |
| ◎ win rate, ◎ top-3 rate (per strategy) | how often the top pick wins / places | the favourite's win and top-3 rate |
| trio hit rate | how often the actual top three is among the five most likely trios | — |
| calibration table and ECE | whether "30%" horses win 30% of the time; ECE is the share-weighted mean gap between predicted and observed rates per probability bin (0 = perfect) | the market's ECE |
| winner log loss | how much probability the winner got | the market's log loss |
| value return | return of the value strategy's flat win bets | about 0.8 (the takeout) |

Where the scores come from:

- **Race-day log**: `aikeiba-evaluate` appends each verified race to
  `data/predictions/log.csv` (one row per race) and `log_runners.csv` (one row per runner,
  used for calibration). `aikeiba-evaluate --summary` prints the running scores.
- **History**: `aikeiba-train` reports the same scores walk-forward by year for the fitted
  weights, the prior weights, the fitted weights with the market, and the market alone.

### How many races are needed

A rate estimated from n races has a standard error of about √(p(1−p)/n). For a top-3 rate
near 60%, the 95% interval is about ±10 points after 100 races and ±5 points after 400. One or
two races say almost nothing; compare strategies and baselines on the log only once it holds
a few hundred races, and on the walk-forward history before that.
