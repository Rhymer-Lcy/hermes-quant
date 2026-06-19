"""Cross-sectional factor definitions and processing.

Convention: every factor is oriented so that HIGHER = more attractive (expected
higher forward return), which makes IC signs and quantile spreads comparable across
factors. All take/return wide daily panels (date x code).

Point-in-time note: value factors (pe/pb) and size (mv) are daily as-of values, so
they carry no announcement lag. Price-based factors (momentum/vol/reversal) use only
past closes. Quality (ROE) will be aligned by announcement date when added.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_xs(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip each row (date) to its [lower, upper] cross-sectional quantiles."""
    lo = df.quantile(lower, axis=1)
    hi = df.quantile(upper, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def zscore_xs(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score each row (mean 0, std 1 across names that date)."""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorize then z-score, cross-sectionally. (Rank IC is invariant to this; it
    matters for the ML combiner and for quantile cut stability.)"""
    return zscore_xs(winsorize_xs(df))


# --- factors (higher = more attractive) ---

def trailing_return(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return close / close.shift(lookback) - 1.0


def momentum(close: pd.DataFrame, lookback: int = 120, skip: int = 20) -> pd.DataFrame:
    """12-1 style: return over [t-lookback, t-skip], skipping the recent `skip` days
    (the short-horizon reversal window)."""
    return close.shift(skip) / close.shift(lookback) - 1.0


def low_vol(close: pd.DataFrame, window: int = 120) -> pd.DataFrame:
    """Negative rolling std of daily returns (low volatility = attractive)."""
    # fill_method=None: do NOT pad across suspensions (pad would inject spurious 0 returns).
    return -close.pct_change(fill_method=None).rolling(window).std()


def earnings_yield(pe_ttm: pd.DataFrame) -> pd.DataFrame:
    """1 / PE_ttm; non-positive PE -> NaN (earnings yield undefined for losses)."""
    return (1.0 / pe_ttm).where(pe_ttm > 0)


def book_yield(pb: pd.DataFrame) -> pd.DataFrame:
    """1 / PB; non-positive PB -> NaN."""
    return (1.0 / pb).where(pb > 0)


def small_size(total_mv: pd.DataFrame) -> pd.DataFrame:
    """Negative log market cap (the small-size premium)."""
    return -np.log(total_mv.where(total_mv > 0))
