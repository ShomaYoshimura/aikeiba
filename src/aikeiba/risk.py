"""Risk sheet: the same fixed risk items scored for every runner (step 2.9 of the workflow).

The point is symmetry. Reasons to oppose a horse are not left to whoever happens to
look for them: each item below is checked for every runner, and a value is recorded
as a number (0 = checked, no risk) or as unknown (None), never guessed.

| item | risk | points |
| --- | --- | --- |
| R1 | previous race x previous finish has a poor record in past editions | -8 / -5 / -2 |
| R2 | distance unproven: poor runs at the distance and/or weak pedigree for it | -6 / -3 |
| R3 | gate: a fancied runner in a gate with a below-average record, or widest gate | -3 / -3 |
| R4 | pedigree vs. today's going (sire's progeny rate in today's condition) | -5 / -2 |
| R5 | running style vs. expected pace: a tag only (the pace shock applies it) | tag |
| R6 | condition: workout rating and body-weight change | -5 / -2, -3 |
| R7 | age: poor record of the age in past editions, cancelled by recent graded form | -3 |
| R8 | class: no top-2 at the race's grade or above | -5 / -3 |

The total is reported but **not added to the log-strengths**: every item overlaps a
factor the model already weighs (``race_trend``, ``first_distance``, ``draw_bias``,
``sire_aptitude``, ``workout_score``, ``class_form`` ...), so adding it would count the
same evidence twice. It is used where the model has no say: the uncertainty of a
runner's score (``sigma_points``, see ``uncertainty.py``), the exclusion rules for
marks and bet axes (``strategies.article_marks``, ``tickets.py``) and the report.
The point values are hand-set priors, not fitted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from aikeiba.factors import last_finish_band
from aikeiba.racecard import WORKOUT_RATINGS, RaceCard, Runner

ITEMS = ("R1", "R2", "R3", "R4", "R6", "R7", "R8")
CAPS = {"R1": -8, "R2": -6, "R3": -4, "R4": -5, "R6": -6, "R7": -3, "R8": -5}
SIGMA_ITEMS = ("R1", "R2", "R8")  # unproven situations: the score itself is less certain
VERDICTS = (  # (upper bound of the total, code, label)
    (-12, "drop", "消し濃厚"),
    (-8, "discount", "割引"),
    (-4, "caution", "軽度警戒"),
    (0, "low", "リスク小"),
)
GRADE_RANK = {"G1": 3, "GI": 3, "Jpn1": 2.5, "G2": 2, "GII": 2, "Jpn2": 1.5, "G3": 1, "GIII": 1}


@dataclass(frozen=True)
class RiskParams:
    min_starts: int = 3  # fewer past-edition starts than this: too few to score R1 / R7
    sire_rate_bad: float = 0.13  # shrunk progeny top-3 rate (prior 0.2) for R4 -5
    sire_rate_weak: float = 0.17  # ... for R4 -2
    sigma_per_item: float = 2.0
    sigma_cap: float = 5.0
    total_cap: float = -20.0
    front_threshold: float = 0.7  # style score from which a runner counts as a front runner
    closer_threshold: float = -0.5


@dataclass
class RunnerRisk:
    number: int
    horse: str
    items: dict  # item -> points (<= 0) or None when unknown
    notes: dict  # item -> short reason in Japanese (for the report)
    r5_tags: list = field(default_factory=list)

    @property
    def total(self) -> float:
        return float(sum(v for v in self.items.values() if v is not None))

    @property
    def capped_total(self) -> float:
        return max(self.total, RiskParams.total_cap)

    @property
    def unknown(self) -> list[str]:
        return [k for k, v in self.items.items() if v is None]

    @property
    def verdict(self) -> tuple[str, str]:
        for bound, code, label in VERDICTS:
            if self.total <= bound:
                return code, label
        return VERDICTS[-1][1:]

    def sigma_points(self, params: RiskParams | None = None) -> float:
        params = params or RiskParams()
        hits = sum(1 for k in SIGMA_ITEMS if (self.items.get(k) or 0) < 0)
        return min(hits * params.sigma_per_item, params.sigma_cap)

    def to_dict(self) -> dict:
        code, label = self.verdict
        return {
            **asdict(self),
            "total": self.total,
            "sigma_points": self.sigma_points(),
            "verdict": code,
            "verdict_label": label,
            "unknown": self.unknown,
        }


def _runs(runner: Runner) -> list:
    return sorted(runner.past_runs, key=lambda p: p.date, reverse=True)


def _rate(rec) -> float:
    return rec.top3 / rec.starts


def popularity_ranks(card: RaceCard) -> np.ndarray | None:
    """Betting rank per runner (1 = favourite) from the card odds, or None without odds."""
    odds = [r.win_odds for r in card.runners]
    if any(o is None for o in odds):
        return None
    return pd.Series(odds).rank(method="min").to_numpy(dtype=int)


def r1_rotation(card: RaceCard, runner: Runner, p: RiskParams):
    runs = _runs(runner)
    if not runs or not runs[0].race_name:
        return None, "前走不明"
    last = runs[0]
    band = last_finish_band(last.finish)
    rec = card.rotation_trends.get(last.race_name, {}).get(band)
    if rec is None:
        return None, f"{last.race_name}{band}着の過去データなし"
    if rec.starts < p.min_starts:
        return 0.0, f"{last.race_name}{band}着は標本{rec.starts}走のみ"
    rate = _rate(rec)
    pts = -8.0 if rate < 0.10 else -5.0 if rate < 0.20 else -2.0 if rate < 0.30 else 0.0
    return pts, f"{last.race_name}{band}着組 複勝率{rate:.0%}（{rec.top3}/{rec.starts}）"


def r2_distance(card: RaceCard, runner: Runner, sire_rate, p: RiskParams):
    runs = [r for r in _runs(runner) if r.distance is not None]
    pedigree_known = sire_rate is not None and not np.isnan(sire_rate)
    if not runs and not pedigree_known:
        return None, "距離実績・血統データなし"
    at_distance = [
        r
        for r in runs
        if abs(r.distance - card.distance) <= 100 and (r.surface in (None, card.surface))
    ]
    if any(r.finish <= 3 for r in at_distance):
        return 0.0, "当該距離で3着内あり"
    poor = bool(at_distance) and all(r.finish > 5 for r in at_distance)
    unfit = pedigree_known and sire_rate < p.sire_rate_weak
    big_change = bool(runs) and abs(runs[0].distance - card.distance) >= 400
    if poor and unfit:
        return -6.0, "当該距離で凡走かつ血統不適"
    if poor:
        return -3.0, "当該距離で凡走のみ"
    if unfit:
        return -3.0, "血統の条件適性が低い"
    if big_change:
        return -3.0, f"前走から{abs(runs[0].distance - card.distance)}mの距離変更で未実証"
    return 0.0, "距離リスクなし"


def _draw(runner: Runner) -> int:
    return runner.draw if runner.draw is not None else runner.number


def r3_draw(card: RaceCard, runner: Runner, popularity, p: RiskParams):
    stats = card.course_draw_stats
    if not stats:
        return None, "枠順データなし"
    starts = sum(r.starts for r in stats.values())
    mean = sum(r.top3 for r in stats.values()) / starts if starts else None
    rec = stats.get(str(_draw(runner)))
    pts, why = 0.0, []
    if popularity is not None and popularity <= 4 and rec and mean and _rate(rec) < mean:
        pts -= 3.0
        why.append(
            f"{popularity}番人気で{_draw(runner)}枠（複勝率{_rate(rec):.0%}<平均{mean:.0%}）"
        )
    inner = [stats[g] for g in stats if g.isdigit() and int(g) <= 3]
    outer = [stats[g] for g in stats if g.isdigit() and int(g) >= 6]
    max_draw = max(_draw(r) for r in card.runners)
    if inner and outer and _draw(runner) == max_draw:
        rin = sum(r.top3 for r in inner) / max(1, sum(r.starts for r in inner))
        rout = sum(r.top3 for r in outer) / max(1, sum(r.starts for r in outer))
        if rin > rout:
            pts -= 3.0
            why.append(f"大外枠で内枠有利（内{rin:.0%}・外{rout:.0%}）")
    if popularity is None and not why:
        return None, "人気未確定（大外判定のみ非該当）"
    return max(pts, CAPS["R3"]), "／".join(why) or "枠順リスクなし"


def r4_pedigree_going(card: RaceCard, sire_rate, p: RiskParams):
    if sire_rate is None or np.isnan(sire_rate):
        return None, "父の条件別成績なし"
    if sire_rate < p.sire_rate_bad:
        return -5.0, f"父産駒の今回条件({card.going})複勝率{sire_rate:.0%}：不適"
    if sire_rate < p.sire_rate_weak:
        return -2.0, f"父産駒の今回条件({card.going})複勝率{sire_rate:.0%}：やや不適"
    return 0.0, f"父産駒の今回条件複勝率{sire_rate:.0%}"


def r6_condition(runner: Runner):
    rating, change = runner.workout_rating, runner.horse_weight_change
    if rating is None and change is None:
        return None, "追い切り評価・馬体重なし"
    pts, why = 0.0, []
    if rating is not None:
        value = WORKOUT_RATINGS[rating]
        if value < 0:
            pts -= 5.0
            why.append(f"追い切り{rating}")
        elif value == 0:
            pts -= 2.0
            why.append(f"追い切り{rating}")
    if change is not None and abs(change) >= 10:
        pts -= 3.0
        why.append(f"馬体重{change:+.0f}kg")
    return max(pts, CAPS["R6"]), "／".join(why) or "状態リスクなし"


def r7_age(card: RaceCard, runner: Runner, r1_note_rate, p: RiskParams):
    if runner.age is None:
        return None, "年齢不明"
    rec = card.trends.get("age", {}).get(str(runner.age))
    if rec is None:
        return None, f"{runner.age}歳の過去データなし"
    if rec.starts < p.min_starts or _rate(rec) >= 0.10:
        return 0.0, f"{runner.age}歳 複勝率{_rate(rec):.0%}（{rec.top3}/{rec.starts}）"
    today = pd.Timestamp(card.date)
    recent_graded = any(
        r.finish <= 3
        and GRADE_RANK.get(r.grade or "", 0) > 0
        and (today - pd.Timestamp(r.date)).days <= 365
        for r in runner.past_runs
    )
    if recent_graded or (r1_note_rate is not None and r1_note_rate > 0.30):
        return 0.0, f"{runner.age}歳は不振だが直近1年の重賞好走で打ち消し"
    return -3.0, f"{runner.age}歳 複勝率{_rate(rec):.0%}（{rec.top3}/{rec.starts}）"


def r8_class(card: RaceCard, runner: Runner):
    target = GRADE_RANK.get(card.grade or "")
    if not target:
        return None, "レース格不明"
    if not runner.past_runs:
        return None, "戦績なし"
    at_level = [r for r in runner.past_runs if GRADE_RANK.get(r.grade or "", 0) >= target]
    if any(r.finish <= 2 for r in at_level):
        return 0.0, f"{card.grade}以上で連対あり"
    if at_level:
        return -3.0, f"{card.grade}以上は出走のみ（最高{min(r.finish for r in at_level)}着）"
    below = [r for r in runner.past_runs if target - 1 <= GRADE_RANK.get(r.grade or "", 0) < target]
    if any(r.finish == 1 for r in below):
        return -3.0, f"{card.grade}初挑戦（一つ下の格で勝利あり）"
    return -5.0, f"{card.grade}初挑戦・格下でも未勝利"


def r5_tags(style: np.ndarray, p: RiskParams) -> list[list[str]]:
    n_front = int((style >= p.front_threshold).sum())
    tags = []
    for s in style:
        t = []
        if n_front >= 3 and s >= p.front_threshold:
            t.append("先行争い激化")
        if n_front <= 1 and s <= p.closer_threshold:
            t.append("スロー濃厚で後方")
        tags.append(t)
    return tags


def assess_risks(
    card: RaceCard,
    raw: pd.DataFrame | None = None,
    style: np.ndarray | None = None,
    params: RiskParams | None = None,
) -> list[RunnerRisk]:
    """Risk sheet for every runner, in card order.

    ``raw`` is the raw factor table of ``strength.evaluate_card`` (for the sire's rate in
    today's condition, from history or card); ``style`` the running-style scores of
    ``simulate.style_scores`` (for the R5 tags).
    """
    p = params or RiskParams()
    pop = popularity_ranks(card)
    sire = (
        raw["sire_aptitude"].to_numpy(dtype=float)
        if raw is not None and "sire_aptitude" in raw
        else [None] * card.field_size
    )
    tags = r5_tags(style, p) if style is not None else [[] for _ in card.runners]
    out = []
    for i, runner in enumerate(card.runners):
        items, notes = {}, {}
        items["R1"], notes["R1"] = r1_rotation(card, runner, p)
        rot_rate = None
        if runner.past_runs and runner.past_runs[0].race_name:
            last = _runs(runner)[0]
            rec = card.rotation_trends.get(last.race_name, {}).get(last_finish_band(last.finish))
            rot_rate = _rate(rec) if rec and rec.starts >= p.min_starts else None
        items["R2"], notes["R2"] = r2_distance(card, runner, sire[i], p)
        items["R3"], notes["R3"] = r3_draw(card, runner, None if pop is None else pop[i], p)
        items["R4"], notes["R4"] = r4_pedigree_going(card, sire[i], p)
        items["R6"], notes["R6"] = r6_condition(runner)
        items["R7"], notes["R7"] = r7_age(card, runner, rot_rate, p)
        items["R8"], notes["R8"] = r8_class(card, runner)
        out.append(RunnerRisk(runner.number, runner.horse, items, notes, tags[i]))
    return out
