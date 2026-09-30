---
name: predict-race
description: Predict a JRA graded race (G1/G2/G3) from its name. Collects entries, horse/jockey/trainer/pedigree data, workouts, stable comments and public prediction trends into a race card, abstracts them into factor vectors, runs a 1,000,000-trial Monte Carlo simulation and reports the top three picks with trio/trifecta probabilities. Use when the user names a race (e.g. "天皇賞（秋）を予想して", "/predict-race 有馬記念").
---

# Predict a graded race

Input: a race name, optionally with a date or year. Reply to the user in Japanese.

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

1. Run the `race-card-collector` agent: race, cutoff, output `<dir>/base.json`.
2. Split the field into groups of about four horses and run one `runner-profile-collector`
   agent per group **in parallel**: race conditions, cutoff, the horses (number, name,
   jockey), output `<dir>/runners-<n>.json`.
3. Merge and validate:

```bash
uv run aikeiba-card merge <dir>/base.json <dir>/runners-*.json -o <dir>/card.json
```

If an agent reports missing fields, you may fill them yourself from a cited source, under
the same rules: nothing published after the cutoff, **never guess or invent a value**, check a
site's terms before fetching it, and add every URL to `sources`. A missing value counts as the
field average. The field list and the comment-scoring rubric are in the agent definitions
(`.claude/agents/`); the card format is in `examples/race_card.example.json`.

## 3. Simulate

```bash
uv run aikeiba-simulate <dir>/card.json --sims 1000000 --json <dir>/prediction.json \
  [--history data/history.parquet] [--weights data/models/weights.json]
```

Pass `--history` and `--weights` when those files exist: statistics then come from the full
history (the card's `stats` and `trends` are only a fallback), and the factor weights are the
ones fitted by `aikeiba-train`. If the card is rejected, fix the card (never the validation)
and run again.

## 4. Report (in Japanese)

1. The race, the data stage, the going, and the sources.
2. Every runner's win, top-2 and top-3 probability.
3. The picks: ◎ (highest win probability), ○ (highest top-2 of the rest), ▲ (highest top-3 of
   the rest). For each, explain the result with the `reasons` in the JSON (the factors that
   moved its score most), translated into plain Japanese, plus any notable comment.
4. The five most likely 馬連, 三連複 and 三連単 combinations.
5. With odds, runners whose win probability × odds exceeds 1.1, as value candidates.
6. Caveats, briefly:
   - Whether the weights are `prior` (hand-set, not validated) or `fitted` (and on how many races).
   - Factors that were unknown for every runner (`missing_factors`).
   - More trials only reduce sampling error; accuracy depends on the inputs.

After the race, the prediction can be scored with the verify-race skill (step 5 there), which
fetches the result and logs it with `aikeiba-evaluate`.
