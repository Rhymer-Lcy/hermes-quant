"""Shared inference helpers for the event studies.

Mirrors the sibling plutus repo's backtest.metrics (the two projects are standalone, so the
~30 lines are ported rather than imported), including the lesson its tests caught there: the
zero-dispersion guard must use a TOLERANCE, because the sample standard deviation of a constant
series is not exactly zero in floating point and an exact `== 0` check would report a t-stat of
1e16 instead of NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def tstat(x: pd.Series) -> float:
    """Plain t-stat of the mean against zero. NaN when there is nothing to test."""
    x = pd.Series(x, dtype=float).dropna()
    if len(x) < 2:
        return float("nan")
    sd = float(x.std(ddof=1))
    if not np.isfinite(sd) or sd < 1e-12:      # tolerance, not == 0: see module docstring
        return float("nan")
    return float(x.mean() / (sd / np.sqrt(len(x))))


def clustered_tstat(x: pd.Series, dates: pd.Series, freq: str = "M") -> float:
    """Clustering-robust t-stat: average the observations inside each calendar period first,
    then take the t-stat across periods.

    Limit-up events cluster violently in time (2015 alone supplies ~29% of this repo's
    limit-up sample), so an event-level t treats hundreds of same-frenzy events as independent
    draws and badly overstates significance. Pre-registered verdicts use THIS statistic."""
    g = pd.DataFrame({"x": np.asarray(x, dtype=float),
                      "p": pd.DatetimeIndex(dates).to_period(freq)}).dropna()
    return tstat(g.groupby("p")["x"].mean())
