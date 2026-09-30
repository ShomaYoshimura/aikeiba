---
name: verify-race
description: Test the prediction system on a JRA graded race that has already been run. Freezes a prediction made only from information published before the race, then fetches the actual result, scores it with aikeiba-evaluate and logs it. Use when the user asks to try, backtest or check the system on a past race (e.g. "先週のスプリンターズSで試して", "/verify-race 天皇賞（秋） 2025").
---

# Verify the system on a past race

Input: a race name and date (or year). Reply to the user in Japanese.

The whole point is a fair test, so the prediction must be built as if on race day. Keep the
roles apart: the collectors never see the result, and the result is fetched only after the
prediction is saved.

## 1. Set up

- Identify the race (name, grade, date, course, post time) without opening its result.
  Work in `data/race_cards/<date>-<slug>/`.
- Cutoff: the evening before race day (23:59 JST), unless the user sets another. Odds and
  horse weights from race day are then not used; say so in the report.
- If you have already seen this race's result (in this conversation, a search snippet, or
  prior knowledge), say so to the user up front, and rely even more strictly on the agents
  and on cited sources for every value.

## 2. Collect and predict (no results)

Follow steps 2 and 3 of the predict-race skill with the cutoff above: the `race-card-collector`
and parallel `runner-profile-collector` agents, `aikeiba-card merge`, then:

```bash
uv run aikeiba-simulate <dir>/card.json --sims 1000000 --seed 0 --json <dir>/prediction.json \
  [--history data/history.parquet] [--weights data/models/weights.json]
```

Keep `--seed 0` so the run can be reproduced.

## 3. Freeze

Do not change `card.json` or `prediction.json` after this point. Record their hashes:

```bash
sha256sum <dir>/card.json <dir>/prediction.json > <dir>/frozen.sha256
```

## 4. Fetch the result

Run the `race-result-checker` agent with output `<dir>/result.json`.

## 5. Score and log

```bash
sha256sum -c <dir>/frozen.sha256
uv run aikeiba-evaluate <dir>/prediction.json <dir>/result.json --log data/predictions/log.csv
```

## 6. Report (in Japanese)

1. Setup: race, cutoff, data stage, weights (`prior` or `fitted`), missing factors, sources,
   and whether the result was already known to you.
2. Prediction vs. result: for each strategy (的中重視 / 両立 / 回収重視), ◎○▲ and their
   finishing positions, next to the favourite's finish (the baseline); every runner's
   predicted win and top-3 probability next to its actual position.
3. Hits: 馬連 / 三連複 / 三連単 (and at what rank in the top five, if at all), and the value
   strategy's return (say whether payouts came from the result or the card odds).
4. Calibration of this race: the winner's predicted probability and model rank, the model's
   log loss, and the same for the market when odds were used.
5. What the model missed: the factors behind the picks (`reasons`) and what distinguished the
   actual top three, from the card only.
6. The running summary printed from the log: hit rates per strategy against the favourite,
   and calibration (ECE and the calibration table). The primary scores are hit rate and
   calibration (`docs/methodology.md`). Stress that one race says little: with ~100 races a
   top-3 rate still has about ±10 points of uncertainty.
