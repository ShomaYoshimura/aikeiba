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

Use WebSearch and WebFetch. Prefer the official JRA site (jra.go.jp). Before fetching from
any other site, check that its terms allow automated access; skip it if they don't.

**Never guess or invent a value.** Leave out any field you could not confirm; a missing value
counts as the field average. List every URL used in `sources`. The full format is in
`examples/race_card.example.json`; `src/aikeiba/racecard.py` validates it.

Race: `name`, `grade`, `date` (YYYY-MM-DD), `course`, `surface` (芝/ダート), `distance` (m),
`going` (良/稍重/重/不良; use the forecast if not announced and say so).

Per runner:

| Area | Fields |
| --- | --- |
| Identity | `number` (馬番), `horse`, `draw` (枠番), `age`, `sex` |
| Connections | `jockey`, `trainer`, `owner`, `breeder`, `region` (産地) |
| Pedigree | `sire`, `damsire` |
| Condition | `weight_carried`, `horse_weight`, `horse_weight_change`, `win_odds` (only if all runners have odds) |
| Style | `running_style`: 逃げ/先行/差し/追込, judged from positions in recent races |
| Past runs | `past_runs`: up to 10, most recent first, each with `date`, `finish`, `field_size`, `race_name`, `course`, `surface`, `distance`, `going`, `grade`, `jockey`, `last3f` |
| Workouts | `workouts`: latest first, `date`, `course` (e.g. 美浦W, 栗東坂路), `time_4f`, `last_1f`, `intensity`; or `workout_rating` A-E if a source grades them |
| Stats | `stats`: `{"starts": n, "top3": k}` records for `jockey_year`, `jockey_graded`, `jockey_course`, `trainer`, `trainer_graded`, `trainer_jockey`, `owner`, `breeder`, `region`, `sire_condition` (sire's progeny on today's surface, distance band and going), `damsire_condition`; and `jockey_h2h` (0-1, share of shared races the jockey finished ahead of this field's other jockeys) |
| Current | `stable_comment` (quote or close paraphrase), `comment_score`, `consensus_share` |

Race-level, from sources such as "過去10年の傾向" pages:

- `trends`: records by `draw` (枠番), `age`, `sex`, `running_style`, `last_finish`
  (`1`, `2-3`, `4-5`, `6+`) over past editions of this race.
- `course_draw_stats`: records by gate for this course, surface and distance over all races.

### Scoring the current information

- `comment_score` (stable comments, 厩舎コメント), from -2 to 2:
  +2 clear, specific confidence ("best condition ever", "aiming at this race");
  +1 positive; 0 routine or vague; -1 hedged or a minor concern (slightly heavy, draw worry);
  -2 a clear problem (injury, missed work, "just a run"). Score only what the comment says,
  not the horse's reputation. Keep the text in `stable_comment` so the user can check it.
- `consensus_share` (online predictions): among the predictions you found, the share that put
  a top mark (◎ or ○) on this horse. Treat it as a weak signal: it mostly repeats the odds
  and popular opinion. Never let it override the data; it has a small weight in the model.
  Use several independent sources, and report how many predictions it is based on.

## 3. Simulate

Write the card to `data/race_cards/<date>-<slug>.json` (the `data/` directory is gitignored).

```bash
uv run aikeiba-simulate data/race_cards/<file>.json --sims 1000000 \
  --json data/race_cards/<file>.result.json \
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
