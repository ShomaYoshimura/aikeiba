# Data sources

The collector agents read public pages through `aikeiba-fetch`, which obeys robots.txt,
waits at least 5 seconds between requests to a host, caches pages under `data/cache/http`
and refuses any site whose terms have not been recorded as reviewed.

## Before using a site

1. Read the site's terms of use and check that automated access for personal analysis is
   allowed (and robots.txt, which the tool also enforces).
2. Record it: `uv run aikeiba-fetch --review-terms <site> --note "<what you checked>"`.
   Only the user does this; the agents must not.
3. Allow the site's domains in the environment's network settings when running in a cloud
   session.

`uv run aikeiba-fetch --sites` lists the sources and their review status.

## What each site is used for

| Site | Domains | Used for | Card fields |
| --- | --- | --- | --- |
| netkeiba | www / race / db / news.netkeiba.com | entries and gates, odds, horse past performances (time, margin, passing order, popularity, last 3F, weight carried), jockey / trainer / sire / dam records, course data | `runners`, `past_runs`, `stats`, `course_trends`, `course_draw_stats`, `win_odds`, `previous_odds` |
| keibalab | www.keibalab.jp | past-edition trends ("過去10年"), course data by gate, running style and sire, entries, odds | `trends`, `course_trends`, `course_draw_stats` |
| keibabook | p.keibabook.co.jp | workouts and stable comments, **free pages only** | `workouts`, `stable_comment` / `comment_score` |
| umanity | umanity.jp | public predictions | `consensus_share` |
| uma-x | uma-x.jp | public (AI) predictions and figures | `consensus_share` |
| note | note.com | free prediction articles | `consensus_share` only |
| jra | www.jra.go.jp | official entries, going, cushion value and moisture, results | `race`, result JSON |

Rules for every site:

- Free, logged-out pages only. Never log in, never read paid or members-only content, never
  work around a paywall.
- Public predictions (umanity, uma-x, note) feed only `consensus_share`; they are a weak
  signal and must not shape any other field.
- Keep the URL of every page used in the card's `sources`.

## Past races and leakage

Pages about a race usually show its result once it has been run. When verifying a past
race (`/verify-race`), the collectors take race-level facts from pages that do not show the
result where possible, take horse data from past-performance pages and keep only runs
before the race date, and report any page that showed the result. The result itself is
fetched only by the race-result-checker, after the prediction is frozen.
