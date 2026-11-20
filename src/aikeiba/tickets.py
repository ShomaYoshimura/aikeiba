"""Budgeted tickets from the report marks, with estimated returns (step 8 of the workflow).

- **Allocation.** About 30% of the budget on ◎ to win, 30% on quinellas around ◎
  (◎-○, ◎-▲, and ○-▲ as the cover for ◎ missing), 40% on trios (◎○▲ and ◎-○ / ◎-▲
  with each △, at most ``max_flow`` points). With ◎ at expected value 2 or more, the
  win share goes up. Within a bet type the stakes are dutched (proportional to
  1 / odds, so every hit returns about the same), rounded to 100 yen.
- **Odds.** Win odds are the card's. Quinella and trio odds are *estimates*: the JRA
  payout rate divided by the market probability of the combination (Harville on the
  odds-implied win probabilities). Without odds there are no estimates.
- **Returns.** Every possible first three is enumerated with Harville on the model's
  win probabilities, giving the expected return, the chance of losing everything and
  the chance of a profit; plus three named outcomes (A: ◎-○-△, B: ◎-▲-△, C: ○-▲-△).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from aikeiba.probability import harville_exacta, harville_trifecta

PAYOUT_RATE = {"win": 0.80, "quinella": 0.775, "trio": 0.75}  # JRA, after takeout
BET_LABELS = {"win": "単勝", "quinella": "馬連", "trio": "3連複"}


@dataclass(frozen=True)
class TicketParams:
    budget: int = 10_000
    shares: tuple[float, float, float] = (0.30, 0.30, 0.40)  # win, quinella, trio
    strong_shares: tuple[float, float, float] = (0.40, 0.25, 0.35)  # ◎ with EV >= strong_ev
    strong_ev: float = 2.0
    max_flow: int = 6  # trio points with a △ in the third column
    unit: int = 100


def _round(x: float, unit: int) -> int:
    return max(unit, int(round(x / unit)) * unit)


def market_estimates(odds) -> tuple[np.ndarray, np.ndarray] | None:
    """Estimated quinella (n x n) and trio (n x n x n, sorted index) odds from win odds."""
    if odds is None:
        return None
    q = 1.0 / np.asarray(odds, float)
    q = q / q.sum()
    ex = harville_exacta(q)
    tri = harville_trifecta(q)
    pair = ex + ex.T
    trio = sum(tri.transpose(p) for p in permutations(range(3)))
    with np.errstate(divide="ignore"):
        return PAYOUT_RATE["quinella"] / pair, PAYOUT_RATE["trio"] / trio


def build_tickets(marks, odds, ev_honmei=None, params: TicketParams | None = None) -> list[dict]:
    """Tickets as dicts with ``bet``, ``runners`` (indices), ``stake`` and ``odds`` (estimate).

    ``marks`` is the list of (mark, index) from ``strategies.article_marks``.
    """
    p = params or TicketParams()
    first = {}
    deltas = []
    for m, i in marks:
        if m in ("◎", "○", "▲") and m not in first:
            first[m] = i
        elif m == "△":
            deltas.append(i)
    if "◎" not in first:
        return []
    est = market_estimates(odds)
    shares = p.strong_shares if ev_honmei is not None and ev_honmei >= p.strong_ev else p.shares
    h, t, a = first["◎"], first.get("○"), first.get("▲")

    win = [("win", (h,))]
    quin = [(("quinella"), tuple(sorted(x))) for x in ((h, t), (h, a), (t, a)) if None not in x]
    trio = []
    if t is not None and a is not None:
        trio.append(("trio", tuple(sorted((h, t, a)))))
    flows = [(h, x, d) for d in deltas for x in (t, a) if x is not None]
    for combo in flows[: p.max_flow]:
        key = ("trio", tuple(sorted(combo)))
        if key not in trio:
            trio.append(key)

    def odds_of(bet, runners):
        if bet == "win":
            return None if odds is None else float(odds[runners[0]])
        if est is None:
            return None
        return float(est[0][runners] if bet == "quinella" else est[1][runners])

    tickets = []
    for share, group in zip(shares, (win, quin, trio), strict=True):
        if not group:
            continue
        o = [odds_of(b, r) for b, r in group]
        weights = (
            np.full(len(group), 1.0 / len(group))
            if any(x is None for x in o)
            else (1.0 / np.asarray(o)) / (1.0 / np.asarray(o)).sum()
        )
        for (bet, runners), w, x in zip(group, weights, o, strict=True):
            tickets.append(
                {
                    "bet": bet,
                    "runners": list(runners),
                    "stake": _round(p.budget * share * w, p.unit),
                    "odds": x,
                }
            )
    return tickets


def payout(tickets: list[dict], order: tuple[int, int, int]) -> dict:
    """Return per bet type for an actual first three ``order`` (runner indices)."""
    out = {"win": 0.0, "quinella": 0.0, "trio": 0.0}
    for tk in tickets:
        r, o = tk["runners"], tk["odds"]
        if o is None:
            continue
        hit = (
            (tk["bet"] == "win" and r[0] == order[0])
            or (tk["bet"] == "quinella" and set(r) == set(order[:2]))
            or (tk["bet"] == "trio" and set(r) == set(order))
        )
        if hit:
            out[tk["bet"]] += tk["stake"] * o
    return out


def evaluate_tickets(tickets: list[dict], win_probs) -> dict:
    """Expected return and loss probabilities under Harville on the model's win rates."""
    stake = sum(t["stake"] for t in tickets)
    if not tickets or any(t["odds"] is None for t in tickets):
        return {"stake": stake, "estimated": False}
    tri = harville_trifecta(np.asarray(win_probs, float))
    n = tri.shape[0]
    exp_ret = p_zero = p_profit = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                pr = tri[i, j, k]
                if pr <= 0:
                    continue
                ret = sum(payout(tickets, (i, j, k)).values())
                exp_ret += pr * ret
                p_zero += pr * (ret == 0)
                p_profit += pr * (ret > stake)
    for t in tickets:
        t["hit_probability"] = _hit_probability(t, tri)
    return {
        "stake": stake,
        "estimated": True,
        "expected_return": exp_ret,
        "expected_multiple": exp_ret / stake if stake else None,
        "p_total_loss": p_zero,
        "p_profit": p_profit,
    }


def _hit_probability(ticket: dict, tri: np.ndarray) -> float:
    r = ticket["runners"]
    if ticket["bet"] == "win":
        return float(tri[r[0]].sum())
    if ticket["bet"] == "quinella":
        a, b = r
        return float(tri[a, b].sum() + tri[b, a].sum())
    return float(sum(tri[x] for x in permutations(r)))


def return_scenarios(marks, tickets: list[dict], win_probs) -> list[dict]:
    """Returns for A (◎-○-△), B (◎-▲-△) and C (○-▲-△, ◎ out of the first three)."""
    first = {}
    for m, i in marks:
        first.setdefault(m, i)
    h, t, a = first.get("◎"), first.get("○"), first.get("▲")
    used = {h, t, a}
    third = first.get("△")
    if third is None:  # the best unmarked runner
        third = next((int(i) for i in np.argsort(-np.asarray(win_probs)) if i not in used), None)
    stake = sum(tk["stake"] for tk in tickets)
    out = []
    for code, label, order in (
        ("A", "本命順当決着（◎-○-△）", (h, t, third)),
        ("B", "◎1着・▲2着の穴本線（◎-▲-△）", (h, a, third)),
        ("C", "◎が圏外、○▲決着の保険（○-▲-△）", (t, a, third)),
    ):
        if None in order:
            continue
        pay = payout(tickets, order)
        total = sum(pay.values())
        out.append(
            {
                "code": code,
                "label": label,
                "order": list(order),
                "returns": pay,
                "total": total,
                "multiple": total / stake if stake else None,
            }
        )
    return out
