---
name: predict-race
description: Predict a JRA race from its name. Collects the entries, jockeys, recent form and going from public sources, writes a race card, runs a 1,000,000-trial Monte Carlo simulation and reports the top three picks with trio/trifecta probabilities. Use when the user names a race (e.g. "天皇賞（秋）を予想して", "/predict-race 有馬記念").
---

# Predict a race

Input: a race name, optionally with a date or year. Reply to the user in Japanese.

## 1. Identify the race

- Resolve the name to one running: the next upcoming one unless the user gives a date.
  Confirm course, surface, distance and date.
- Entries are published in stages. Nominations (特別登録) come out about a week ahead; the
  final field with gates (枠順確定) usually comes out two days before the race; odds only exist
  once betting opens. Tell the user which stage the data is at. If the final field is not
  out, simulate the nominated horses and say that gates and odds are missing.

## 2. Collect the race card

Use WebSearch and WebFetch. Prefer the official JRA site (jra.go.jp). Before fetching
from any other site, check that its terms allow automated access; skip it if they don't.

For each runner, collect only what a source states:

| Field | Meaning |
| --- | --- |
| `number`, `horse` | horse number (馬番) and name. Required |
| `draw` | gate (枠番) |
| `jockey` | jockey name |
| `win_odds` | current decimal win odds (単勝オッズ). Include only if every runner has odds |
| `running_style` | 逃げ / 先行 / 差し / 追込, judged from positions in recent races |
| `recent_finishes` | finishing positions, most recent first, up to 5 |
| `jockey_win_rate` | the jockey's win rate this year, as a fraction |
| `going_record` | `{"starts": n, "top3": k}` on today's going |
| `course_record` | `{"starts": n, "top3": k}` at this course and distance |

Race fields: `name`, `date` (YYYY-MM-DD), `course`, `surface` (`turf`/`dirt`), `distance`
(metres), `going` (良 / 稍重 / 重 / 不良). If the going is not announced yet, use the forecast
and say so.

**Never guess or invent a value.** Leave out any field you could not confirm; the model
treats a missing field as the field average. List every URL you used in `sources`.

Write the card to `data/race_cards/<date>-<slug>.json` (the `data/` directory is gitignored).
`examples/race_card.example.json` shows the format.

## 3. Simulate

```bash
uv run aikeiba-simulate data/race_cards/<file>.json --sims 1000000 --json data/race_cards/<file>.result.json
```

If the card is rejected, fix the card (never the validation) and run again.

## 4. Report (in Japanese)

1. The race, the data stage (nominations / final field / odds), the going, and the sources.
2. A table of every runner: win, top-2 and top-3 probability.
3. The picks: ◎ (highest win probability), ○ (highest top-2 probability of the rest),
   ▲ (highest top-3 probability of the rest), each with one line on why, using only the
   collected data.
4. The five most likely 馬連 (quinella), 三連複 (trio) and 三連単 (trifecta) combinations.
5. When odds exist, runners whose win probability × odds exceeds 1.1, as value candidates.
6. Caveats, stated briefly: the strength weights and shock sizes are provisional priors that
   have not been fitted to data, so the probabilities are not validated. More trials only
   reduce sampling error (the win-probability standard error is already about 0.05% at
   1,000,000 trials); accuracy depends on the inputs. List any fields that were missing.
