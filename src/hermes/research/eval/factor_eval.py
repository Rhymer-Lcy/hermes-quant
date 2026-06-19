"""Single-factor evaluation: rank IC and quantile (layered) forward returns.

This is the rigorous gate BEFORE any ML: does a factor actually predict forward
returns out-of-sample? Computed on NON-OVERLAPPING month-end dates (signal at t,
return t -> next eval date) over the point-in-time index universe, so there is no
look-ahead and no overlapping-window t-stat inflation.

rank IC(t) = Spearman(factor(t), forward_return(t)) across the universe that date.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ICResult:
    ic: pd.Series          # rank IC per eval date
    mean_ic: float
    ic_ir: float           # mean / std (information ratio of the IC series)
    t_stat: float          # ic_ir * sqrt(n_periods)
    hit_rate: float        # fraction of periods with IC > 0
    n_periods: int


def _restrict(f: pd.Series, r: pd.Series, members: set | None):
    df = pd.concat([f, r], axis=1, keys=["f", "r"]).dropna()
    if members is not None:
        df = df.loc[df.index.isin(members)]
    return df


def compute_ic(factor: pd.DataFrame, close: pd.DataFrame, eval_dates: list,
               members_asof=None) -> ICResult:
    fwd = close.reindex(eval_dates)
    ic = {}
    for i in range(len(eval_dates) - 1):
        t, t1 = eval_dates[i], eval_dates[i + 1]
        r = fwd.loc[t1] / fwd.loc[t] - 1.0
        f = factor.loc[t]
        df = _restrict(f, r, members_asof(t) if members_asof else None)
        if len(df) >= 5:
            ic[t] = df["f"].corr(df["r"], method="spearman")
    ic = pd.Series(ic).dropna()
    mean_ic = float(ic.mean()) if len(ic) else float("nan")
    sd = float(ic.std()) if len(ic) else float("nan")
    ir = mean_ic / sd if sd and sd > 0 else float("nan")
    t_stat = ir * np.sqrt(len(ic)) if len(ic) and np.isfinite(ir) else float("nan")
    hit = float((ic > 0).mean()) if len(ic) else float("nan")
    return ICResult(ic, mean_ic, float(ir), float(t_stat), hit, len(ic))


def quantile_returns(factor: pd.DataFrame, close: pd.DataFrame, eval_dates: list,
                     n_q: int = 5, members_asof=None) -> pd.Series:
    """Mean forward return per factor quantile (0 = lowest factor .. n_q-1 = highest),
    averaged over eval dates. A monotone increasing profile + positive top-minus-bottom
    spread indicates a usable factor."""
    fwd = close.reindex(eval_dates)
    rows = []
    for i in range(len(eval_dates) - 1):
        t, t1 = eval_dates[i], eval_dates[i + 1]
        r = fwd.loc[t1] / fwd.loc[t] - 1.0
        df = _restrict(factor.loc[t], r, members_asof(t) if members_asof else None)
        if len(df) < n_q:
            continue
        q = pd.qcut(df["f"].rank(method="first"), n_q, labels=False)
        rows.append(df["r"].groupby(q).mean())
    return pd.DataFrame(rows).mean() if rows else pd.Series(dtype=float)
