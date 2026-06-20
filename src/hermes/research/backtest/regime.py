"""Market-regime exposure: a broad-index trend filter to cut systematic drawdown.

A-share drawdowns are largely systematic (the whole market falls together). Gating
gross exposure by a simple index trend -- above vs below its long moving average --
sidesteps the worst of those drawdowns at the cost of some upside and some whipsaw.
"""
from __future__ import annotations

from typing import Any, Callable

import pandas as pd


def trend_exposure(index_close: pd.Series, ma_window: int = 200) -> pd.Series:
    """1.0 when the index is at/above its `ma_window`-day MA (risk-on), else 0.0.

    NaN during the MA warm-up so exposure_lookup defaults to fully invested there.
    Uses only past closes (the MA at date t includes t), so it is point-in-time."""
    ma = index_close.rolling(ma_window).mean()
    return (index_close >= ma).astype(float).where(ma.notna())


def exposure_lookup(exposure: pd.Series) -> Callable[[Any], float]:
    """f(date) -> exposure as of the latest available date <= the query date; defaults
    to 1.0 (no filter) before the series starts / during the MA warm-up."""
    exp = exposure.dropna().sort_index()

    def asof(when) -> float:
        s = exp.loc[:pd.Timestamp(when)]
        return float(s.iloc[-1]) if len(s) else 1.0

    return asof
