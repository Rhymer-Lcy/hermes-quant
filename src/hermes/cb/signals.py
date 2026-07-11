"""Shared CB signal construction -- ONE implementation for the study AND the paper ledger,
so the served signal can never drift from the researched one (the live/strategy.py
principle, applied to the CB line).

Frozen in docs/cb_lake.md before any result existed: the double-low score is the Eastmoney
close plus the conversion premium (percentage points); a bond is eligible at a signal date
when it traded that day, has >= MIN_HISTORY_DAYS prior traded days, its 20-day median
turnover (close x volume, >= 10 traded days in the window) clears its exchange's floor,
and the score inputs exist. Pricing/trading always uses the Sina panel.
"""
from __future__ import annotations

import pandas as pd

MIN_HISTORY_DAYS = 60
TURNOVER_WINDOW = 20
TURNOVER_MIN_PERIODS = 10


def panels(bars: pd.DataFrame, prem: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Wide date x code panels from the long lake frames, on the Sina trading calendar."""
    close = bars.pivot(index="date", columns="code", values="close").sort_index()
    volume = bars.pivot(index="date", columns="code", values="volume").sort_index()
    em_close = prem.pivot(index="date", columns="code", values="close").reindex(close.index)
    em_premium = (prem.pivot(index="date", columns="code", values="conv_premium")
                  .reindex(close.index))
    return {"close": close, "volume": volume, "em_close": em_close, "em_premium": em_premium}


def month_end_signals(calendar: pd.DatetimeIndex, start: str) -> pd.DatetimeIndex:
    """The last trading day of each month in `calendar`, from `start` on."""
    ends = pd.Series(calendar, index=calendar).groupby(calendar.to_period("M")).max()
    return pd.DatetimeIndex(ends[ends >= start])


def turnover_metric(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """The liquidity gate's metric: rolling 20-day median of close x volume."""
    return (close * volume).rolling(TURNOVER_WINDOW, min_periods=TURNOVER_MIN_PERIODS).median()


def base_and_score(close: pd.DataFrame, em_close: pd.DataFrame,
                   em_premium: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(base, score_all): the pre-floor eligibility gates (traded today, enough history,
    score inputs present) and the raw double-low score panel."""
    history_ok = close.notna().cumsum().shift(1) >= MIN_HISTORY_DAYS
    score_all = em_close + em_premium
    return close.notna() & history_ok & score_all.notna(), score_all


def apply_floor(base: pd.DataFrame, turnover: pd.DataFrame,
                floors: dict[str, float]) -> pd.DataFrame:
    """Final eligibility: `base` AND the turnover metric at/above the exchange's floor
    (`floors` maps the code prefix, '11' SH / '12' SZ, to its calibrated floor)."""
    floor_row = pd.Series({c: floors[c[:2]] for c in base.columns})
    return base & turnover.ge(floor_row, axis=1)
