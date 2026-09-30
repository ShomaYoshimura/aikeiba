"""Shrunk rates and entity statistics computed from a history table.

Every rate is shrunk toward a prior with empirical-Bayes style pseudo-counts, so an
entity with three starts does not get an extreme rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def shrunk_rate(successes, trials, prior: float, strength: float):
    """(successes + prior * strength) / (trials + strength); works on scalars and arrays."""
    return (np.asarray(successes, dtype=float) + prior * strength) / (
        np.asarray(trials, dtype=float) + strength
    )


def top3_rate(rows: pd.DataFrame, prior: float, strength: float) -> float:
    """Shrunk top-3 rate of ``rows`` (NaN when there are none, meaning "unknown")."""
    if rows.empty:
        return np.nan
    return float(shrunk_rate(rows["top3"].sum(), len(rows), prior, strength))


def grouped_top3_rates(
    rows: pd.DataFrame, keys: list[str], prior: float, strength: float
) -> pd.Series:
    g = rows.groupby(keys, observed=True)["top3"].agg(["sum", "size"])
    return pd.Series(shrunk_rate(g["sum"], g["size"], prior, strength), index=g.index)


def profile_matrix(rows: pd.DataFrame, entity_col: str, strength: float = 30.0) -> pd.DataFrame:
    """Entity x condition-bin matrix of shrunk top-3 rates: the entity's profile vector.

    Each bin is shrunk toward that bin's overall rate, so a sire with no runners on
    heavy dirt gets the average there rather than an unknown.
    """
    bin_prior = rows.groupby("cond_bin")["top3"].mean()
    g = rows.groupby([entity_col, "cond_bin"], observed=True)["top3"].agg(["sum", "size"])
    prior = bin_prior.reindex(g.index.get_level_values("cond_bin")).to_numpy()
    rates = (g["sum"].to_numpy() + prior * strength) / (g["size"].to_numpy() + strength)
    return pd.Series(rates, index=g.index).unstack("cond_bin")


def head_to_head(rows: pd.DataFrame, entity_col: str, field: list, strength: float = 10.0):
    """For each entity in ``field``: mean over the other entities in the field of the shrunk
    share of shared races in which it finished ahead. 0.5 means even; NaN when unknown.
    """
    sub = rows.loc[rows[entity_col].isin(field), ["race_id", entity_col, "finish_position"]]
    pairs = sub.merge(sub, on="race_id", suffixes=("", "_opp"))
    pairs = pairs[pairs[entity_col] != pairs[f"{entity_col}_opp"]]
    if pairs.empty:
        return {e: np.nan for e in field}
    pairs = pairs.assign(ahead=(pairs["finish_position"] < pairs["finish_position_opp"]))
    g = pairs.groupby([entity_col, f"{entity_col}_opp"])["ahead"].agg(["sum", "size"])
    share = pd.Series(shrunk_rate(g["sum"], g["size"], 0.5, strength), index=g.index)
    out = {}
    for e in field:
        opponents = [o for o in field if o != e]
        vals = [share.get((e, o), np.nan) for o in opponents]
        known = [v for v in vals if not np.isnan(v)]
        out[e] = float(np.mean(known)) if known else np.nan
    return out
