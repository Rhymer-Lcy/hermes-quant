"""Lake cross-checks the study must run BEFORE its result is read (docs/cb_lake.md).

Two independent-source consistency checks, both reported as rates, never silently:
  1. price check -- Eastmoney's value-analysis close vs Sina's bar close per (bond, date):
     two scrapers of two different upstreams should agree to the tick;
  2. revision check -- Eastmoney's conversion-value series must JUMP by about old/new on
     each JSL downward-revision effective date (conversion value = 100 x stock price /
     conversion price, and a >=10% revision dwarfs daily stock noise).
"""
from __future__ import annotations

import pandas as pd


def close_mismatch(bars: pd.DataFrame, premium: pd.DataFrame, tol: float = 0.005) -> dict:
    """Share of joined (bond, date) rows where the two closes disagree by more than `tol`
    (relative). Long inputs: bars[code, date, close] (Sina), premium[code, date, close]
    (Eastmoney). Returns {rows, mismatch_rate, worst}."""
    m = bars[["code", "date", "close"]].merge(
        premium[["code", "date", "close"]], on=["code", "date"], suffixes=("_sina", "_em"))
    rel = (m["close_em"] - m["close_sina"]).abs() / m["close_sina"]
    return {"rows": int(len(m)), "mismatch_rate": float((rel > tol).mean()) if len(m) else 1.0,
            "worst": float(rel.max()) if len(m) else float("nan")}


def revision_jump_matched(conv_value: pd.Series, old_price: float, new_price: float,
                          effective: pd.Timestamp, window: int = 3,
                          frac: float = 0.5) -> bool | None:
    """Does `conv_value` (date-indexed, one bond) jump by at least `frac` of the expected
    old/new - 1 within +-`window` trading days of the revision's effective date? None when
    the event cannot be resolved: bad prices, or the date falls outside the served series
    (an unresolved event is excluded from the match-rate denominator, not a mismatch)."""
    if not (old_price and new_price) or old_price <= new_price:
        return None
    cv = conv_value.dropna().sort_index()
    if len(cv) < 2:
        return None
    pos = int(cv.index.searchsorted(effective))
    if pos == 0 or pos >= len(cv):
        return None
    lo, hi = max(pos - window, 1), min(pos + window + 1, len(cv))
    ratios = cv.iloc[lo - 1:hi].pct_change().dropna()
    if ratios.empty:
        return None
    expected = old_price / new_price - 1.0
    return bool((ratios >= frac * expected).any())
