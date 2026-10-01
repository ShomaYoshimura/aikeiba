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

from aikeiba.model import ModelWeights
from aikeiba.racecard import RaceCard, load_race_card
from aikeiba.strategies import StrategyParams, make_picks
from aikeiba.strength import CardEvaluation, evaluate_card

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
    front_runner_threshold: float = 0.7  # early-speed score from which a runner counts as front
    rain_bias_multiplier: float = 1.3  # wider track-bias spread in rain or snow
    wind_bias_per_ms: float = 0.03  # +3% track-bias spread per m/s of wind


def style_scores(card: RaceCard, early_speed: np.ndarray | None = None) -> np.ndarray:
    """Running-style score per runner (1 = leads, -1 = comes from last): the measured early
    speed from past runs where known, else the running-style label, else 0."""
    labels = np.array([STYLE_SCORE.get(r.running_style, 0.0) for r in card.runners])
    if early_speed is None:
        return labels
    early = np.asarray(early_speed, dtype=float)
    return np.where(np.isnan(early), labels, early)


def race_shocks(
    card: RaceCard,
    n_sims: int,
    rng: np.random.Generator,
    params: ShockParams,
    n_runs: np.ndarray,
    early_speed: np.ndarray | None = None,
) -> np.ndarray:
    n = card.field_size
    style = style_scores(card, early_speed)
    n_front = int((style >= params.front_runner_threshold).sum())
    pace = rng.normal(params.pace_per_front_runner * (n_front - 1), 1.0, (n_sims, 1))
    shocks = -params.pace_effect * pace * style

    draw = np.array([r.draw if r.draw is not None else r.number for r in card.runners], float)
    span = draw.max() - draw.min()
    position = 2 * (draw - draw.min()) / span - 1 if span else np.zeros(n)  # -1 inside, +1 out
    bias_sd = params.track_bias_sd[card.going]
    if card.weather in ("rain", "snow"):
        bias_sd *= params.rain_bias_multiplier
    if card.wind_speed:
        bias_sd *= 1.0 + params.wind_bias_per_ms * card.wind_speed
    bias = rng.normal(0.0, bias_sd, (n_sims, 1))
    shocks -= bias * position  # bias > 0 favours the inside

    sigma = params.unknown_form_sd / np.sqrt(1.0 + n_runs)
    return shocks + rng.normal(size=(n_sims, n)) * sigma


@dataclass
class SimulationResult:
    card: RaceCard
    evaluation: CardEvaluation
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
        if self.evaluation.log_market is not None:
            df["market_win"] = np.exp(self.evaluation.log_market)
            df["win_odds"] = self.odds
            df["ev"] = self.win * self.odds
        return df.sort_values("win", ascending=False, ignore_index=True)

    @property
    def odds(self) -> np.ndarray | None:
        odds = [r.win_odds for r in self.card.runners]
        return None if any(o is None for o in odds) else np.asarray(odds, dtype=float)

    def strategy_picks(self, params: StrategyParams | None = None) -> dict:
        """Marks for each strategy in ``strategies.STRATEGIES``."""
        return make_picks(self.win, self.top2, self.top3, self.odds, params)

    def picks(self) -> list[int]:
        """Runner indices for ◎ ○ ▲ of the hit strategy."""
        return [i for _, i, _, _ in self.strategy_picks()["hit"]]

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
            "graded": self.card.is_graded,
            "n_sims": self.n_sims,
            "weights": "fitted" if self.evaluation.weights.fitted else "prior",
            "uses_market": self.evaluation.log_market is not None,
            "runners": self.table().to_dict(orient="records"),
            "picks": [
                {
                    "mark": m,
                    "number": self.card.runners[i].number,
                    "horse": names[i],
                    "reasons": self.evaluation.reasons(i),
                }
                for m, i in zip(("◎", "○", "▲"), self.picks(), strict=True)
            ],
            "strategies": {
                name: [
                    {
                        "mark": mark,
                        "number": self.card.runners[i].number,
                        "horse": names[i],
                        "win": float(self.win[i]),
                        "top3": float(self.top3[i]),
                        "ev": ev,
                        "stake": stake,
                        "reasons": self.evaluation.reasons(i),
                    }
                    for mark, i, ev, stake in picks
                ]
                for name, picks in self.strategy_picks().items()
            },
            "missing_factors": self.evaluation.missing_factors(),
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
    history: pd.DataFrame | None = None,
    weights: ModelWeights | None = None,
    shocks: bool = True,
    params: ShockParams | None = None,
    chunk_size: int = 100_000,
) -> SimulationResult:
    n = card.field_size
    if n < 3:
        raise ValueError("simulation needs at least three runners")
    params = params or ShockParams()
    rng = np.random.default_rng(seed)
    evaluation = evaluate_card(card, history, weights)
    log_s = evaluation.log_s
    pos_counts = np.zeros((3, n))
    quinella = np.zeros(n * n)
    trio = np.zeros(n**3)
    trifecta = np.zeros(n**3)

    for start in range(0, n_sims, chunk_size):
        m = min(chunk_size, n_sims - start)
        keys = log_s + rng.gumbel(size=(m, n))
        if shocks:
            keys += race_shocks(
                card, m, rng, params, evaluation.n_runs, evaluation.raw["early_speed"].to_numpy()
            )
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
    return SimulationResult(
        card, evaluation, n_sims, cum[0], cum[1], cum[2], quinella, trio, trifecta
    )


def _format(result: SimulationResult) -> str:
    card = result.card
    ev = result.evaluation
    lines = [
        f"{card.name} ({card.grade or 'grade unknown'})  {card.date}  {card.course} "
        f"{card.surface} {card.distance}m  going={card.going}  sims={result.n_sims:,}",
        f"weights={'fitted' if ev.weights.fitted else 'prior'}  "
        f"market={'yes' if ev.log_market is not None else 'no'}",
    ]
    if not card.is_graded:
        lines.append("warning: this system targets graded races (G1/G2/G3)")
    lines.append("")
    with pd.option_context("display.width", 120, "display.float_format", "{:.4f}".format):
        lines.append(result.table().to_string(index=False))
    lines.append("")
    labels = {"hit": "hit (的中重視)", "balanced": "balanced (両立)", "value": "value (回収重視)"}
    for name, picks in result.strategy_picks().items():
        lines.append(f"{labels[name]}:")
        if not picks:
            lines.append("  (no runner above the expected-value threshold, or no odds)")
        for mark, i, exp_value, stake in picks:
            r = card.runners[i]
            why = ", ".join(f"{f} {c:+.2f}" for f, c in ev.reasons(i))
            extra = "" if exp_value is None else f"  EV={exp_value:.2f}"
            extra += "" if stake is None else f"  stake={stake:.1%}"
            lines.append(f"  {mark} {r.number:>2} {r.horse}{extra}  [{why}]")
    for kind in ("quinella", "trio", "trifecta"):
        lines.append(f"\nTop {kind}:")
        for combo, p in result.top_combinations(kind):
            lines.append(f"  {'-'.join(map(str, combo)):<10} {p:.4f}")
    missing = ev.missing_factors()
    if missing:
        lines.append(f"\nfactors unknown for every runner: {', '.join(missing)}")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo simulation of a race card")
    parser.add_argument("card", help="race card JSON")
    parser.add_argument("--sims", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shocks", action="store_true", help="pure Plackett-Luce draws")
    parser.add_argument("--history", help="history parquet to compute statistics from")
    parser.add_argument("--weights", help="fitted weights JSON from aikeiba-train")
    parser.add_argument("--json", help="also write the result as JSON to this path")
    args = parser.parse_args(argv)

    result = simulate_race(
        load_race_card(args.card),
        args.sims,
        args.seed,
        history=pd.read_parquet(args.history) if args.history else None,
        weights=ModelWeights.load(args.weights) if args.weights else None,
        shocks=not args.no_shocks,
    )
    print(_format(result))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2, default=int)


if __name__ == "__main__":
    main()
