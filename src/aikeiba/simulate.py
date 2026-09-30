"""Monte Carlo simulation of one race card.

Each trial draws a finishing order from Plackett-Luce (Gumbel noise on the
log-strengths) plus race-level shocks shared by all runners in that trial:

- pace: more front runners raise the expected pace; a fast pace hurts front runners
  and helps closers, a slow pace the reverse.
- track bias: an inside/outside bias whose spread grows as the going softens.
- form uncertainty: extra per-runner noise that shrinks with the number of known runs.

With ``shocks=False`` the draws are exact Plackett-Luce samples, so win rates converge
to the softmax of the log-strengths (the market probabilities when only odds are known).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aikeiba.racecard import RaceCard, load_race_card
from aikeiba.strength import log_strengths, runs_known

STYLE_SCORE = {"front": 1.0, "stalker": 0.5, "midfield": -0.25, "closer": -1.0}


@dataclass(frozen=True)
class ShockParams:
    """Provisional priors in log-strength units; to be fitted once real data exists."""

    pace_effect: float = 0.15  # shift per 1 sd of pace per unit of running-style score
    pace_per_front_runner: float = 0.5  # expected pace (sd units) per front runner beyond one
    track_bias_sd: dict = field(
        default_factory=lambda: {"good": 0.10, "yielding": 0.15, "soft": 0.20, "heavy": 0.25}
    )
    unknown_form_sd: float = 0.5  # extra noise for a runner with no known runs


def race_shocks(
    card: RaceCard, n_sims: int, rng: np.random.Generator, params: ShockParams
) -> np.ndarray:
    n = card.field_size
    style = np.array([STYLE_SCORE.get(r.running_style, 0.0) for r in card.runners])
    n_front = sum(r.running_style == "front" for r in card.runners)
    pace = rng.normal(params.pace_per_front_runner * (n_front - 1), 1.0, (n_sims, 1))
    shocks = -params.pace_effect * pace * style

    draw = np.array([r.draw if r.draw is not None else r.number for r in card.runners], float)
    span = draw.max() - draw.min()
    position = 2 * (draw - draw.min()) / span - 1 if span else np.zeros(n)  # -1 inside, +1 out
    bias = rng.normal(0.0, params.track_bias_sd[card.going], (n_sims, 1))
    shocks -= bias * position  # bias > 0 favours the inside

    sigma = params.unknown_form_sd / np.sqrt(1.0 + runs_known(card))
    return shocks + rng.normal(size=(n_sims, n)) * sigma


@dataclass
class SimulationResult:
    card: RaceCard
    n_sims: int
    win: np.ndarray
    top2: np.ndarray
    top3: np.ndarray
    quinella: np.ndarray  # counts indexed by a * n + b with a < b
    trio: np.ndarray  # counts indexed by a * n^2 + b * n + c with a < b < c
    trifecta: np.ndarray  # counts indexed by first * n^2 + second * n + third

    def standard_error(self, p: np.ndarray) -> np.ndarray:
        return np.sqrt(p * (1 - p) / self.n_sims)

    def table(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "number": [r.number for r in self.card.runners],
                "horse": [r.horse for r in self.card.runners],
                "jockey": [r.jockey for r in self.card.runners],
                "win": self.win,
                "top2": self.top2,
                "top3": self.top3,
                "win_se": self.standard_error(self.win),
            }
        )
        return df.sort_values("win", ascending=False, ignore_index=True)

    def picks(self) -> list[int]:
        """Indices for ◎ (best win), ○ (best top-2 of the rest), ▲ (best top-3 of the rest)."""
        chosen: list[int] = []
        for p in (self.win, self.top2, self.top3):
            masked = p.copy()
            masked[chosen] = -1.0
            chosen.append(int(np.argmax(masked)))
        return chosen

    def top_combinations(self, kind: str, k: int = 5) -> list[tuple[tuple[int, ...], float]]:
        """Most frequent combinations as (horse numbers, probability)."""
        counts, size = {
            "quinella": (self.quinella, 2),
            "trio": (self.trio, 3),
            "trifecta": (self.trifecta, 3),
        }[kind]
        n = self.card.field_size
        numbers = [r.number for r in self.card.runners]
        out = []
        for key in np.argsort(-counts)[:k]:
            idx = np.unravel_index(key, (n,) * size)
            out.append((tuple(numbers[i] for i in idx), counts[key] / self.n_sims))
        return out

    def to_dict(self) -> dict:
        names = [r.horse for r in self.card.runners]
        return {
            "race": self.card.name,
            "date": self.card.date,
            "going": self.card.going,
            "n_sims": self.n_sims,
            "runners": self.table().to_dict(orient="records"),
            "picks": [
                {"mark": m, "number": self.card.runners[i].number, "horse": names[i]}
                for m, i in zip(("◎", "○", "▲"), self.picks(), strict=True)
            ],
            **{
                kind: [{"numbers": c, "probability": p} for c, p in self.top_combinations(kind)]
                for kind in ("quinella", "trio", "trifecta")
            },
        }


def simulate_race(
    card: RaceCard,
    n_sims: int = 1_000_000,
    seed: int | None = None,
    *,
    shocks: bool = True,
    params: ShockParams | None = None,
    chunk_size: int = 100_000,
) -> SimulationResult:
    n = card.field_size
    if n < 3:
        raise ValueError("simulation needs at least three runners")
    params = params or ShockParams()
    rng = np.random.default_rng(seed)
    log_s = log_strengths(card)
    pos_counts = np.zeros((3, n))
    quinella = np.zeros(n * n)
    trio = np.zeros(n**3)
    trifecta = np.zeros(n**3)

    for start in range(0, n_sims, chunk_size):
        m = min(chunk_size, n_sims - start)
        keys = log_s + rng.gumbel(size=(m, n))
        if shocks:
            keys += race_shocks(card, m, rng, params)
        top = np.argpartition(-keys, 2, axis=1)[:, :3]
        order = np.argsort(-np.take_along_axis(keys, top, axis=1), axis=1)
        top = np.take_along_axis(top, order, axis=1)  # first, second, third
        for k in range(3):
            pos_counts[k] += np.bincount(top[:, k], minlength=n)
        pair = np.sort(top[:, :2], axis=1)
        quinella += np.bincount(pair[:, 0] * n + pair[:, 1], minlength=n * n)
        t = np.sort(top, axis=1)
        trio += np.bincount((t[:, 0] * n + t[:, 1]) * n + t[:, 2], minlength=n**3)
        trifecta += np.bincount((top[:, 0] * n + top[:, 1]) * n + top[:, 2], minlength=n**3)

    cum = np.cumsum(pos_counts, axis=0) / n_sims
    return SimulationResult(card, n_sims, cum[0], cum[1], cum[2], quinella, trio, trifecta)


def _format(result: SimulationResult) -> str:
    card = result.card
    lines = [
        f"{card.name}  {card.date}  {card.course} {card.surface} {card.distance}m  "
        f"going={card.going}  sims={result.n_sims:,}",
        "",
    ]
    with pd.option_context("display.width", 120, "display.float_format", "{:.4f}".format):
        lines.append(result.table().to_string(index=False))
    lines.append("")
    for mark, i in zip(("◎", "○", "▲"), result.picks(), strict=True):
        r = card.runners[i]
        lines.append(f"{mark} {r.number:>2} {r.horse}")
    for kind in ("quinella", "trio", "trifecta"):
        lines.append(f"\nTop {kind}:")
        for combo, p in result.top_combinations(kind):
            lines.append(f"  {'-'.join(map(str, combo)):<10} {p:.4f}")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo simulation of a race card")
    parser.add_argument("card", help="race card JSON")
    parser.add_argument("--sims", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shocks", action="store_true", help="pure Plackett-Luce draws")
    parser.add_argument("--json", help="also write the result as JSON to this path")
    args = parser.parse_args(argv)

    result = simulate_race(
        load_race_card(args.card), args.sims, args.seed, shocks=not args.no_shocks
    )
    print(_format(result))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2, default=int)


if __name__ == "__main__":
    main()
