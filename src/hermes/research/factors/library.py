"""Cross-sectional factor definitions and processing.

Convention: every factor is oriented so that HIGHER = more attractive (expected
higher forward return), which makes IC signs and quantile spreads comparable across
factors. All take/return wide daily panels (date x code).

Point-in-time note: value factors (pe/pb) are daily as-of values, so they carry no
announcement lag; price-based factors (momentum/vol/reversal) use only past closes.
Size (market cap) is implemented in small_size() but its data is NOT yet ingested
(see scripts/ingest_size.py); quality (ROE) will be aligned by announcement date when added.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def restrict_to_universe(panel: pd.DataFrame, members_asof) -> pd.DataFrame:
    """NaN out, per date, every name NOT in the point-in-time universe `members_asof(date)`.

    CRITICAL for survivorship-free studies: a cross-sectional op (winsorize/z-score/blend)
    computed over the survivorship-defined UNION (names ever in the index) leaks future
    membership into the normalization -- a member's standardized score then depends on the
    presence of names that are in the panel only because they JOIN the index LATER. That
    inflates results (observed: an A2 low-vol blend's Calmar fell 0.32 -> 0.28 once fixed).
    Always restrict to the PIT universe BEFORE standardize()/blend() in a PIT study.
    `members_asof`: callable(date)->set[code]."""
    mask = pd.DataFrame(False, index=panel.index, columns=panel.columns)
    for d in panel.index:
        present = panel.columns[panel.columns.isin(members_asof(d))]
        mask.loc[d, present] = True
    return panel.where(mask)


def winsorize_xs(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip each row (date) to its [lower, upper] cross-sectional quantiles.

    NOTE: the cross-section is whatever columns are non-NaN that date. In a PIT study,
    restrict_to_universe() the panel FIRST, or the union leaks (see that function)."""
    lo = df.quantile(lower, axis=1)
    hi = df.quantile(upper, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def zscore_xs(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score each row (mean 0, std 1 across names that date). See the
    survivorship caveat on winsorize_xs / restrict_to_universe."""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorize then z-score, cross-sectionally. (Rank IC is invariant to this; it
    matters for the ML combiner and for quantile cut stability.) In a PIT study, the input
    must already be restrict_to_universe()'d -- the union cross-section leaks otherwise."""
    return zscore_xs(winsorize_xs(df))


def blend(panels: list[pd.DataFrame], weights: list[float] | None = None) -> pd.DataFrame:
    """Combine factor panels into one score: standardize each cross-sectionally (so
    different scales are comparable), then take the weighted mean across factors per
    (date, code), skipping factors missing for that name (a thin-history name still gets
    scored on its available factors rather than dropped). All inputs must already be
    oriented higher = more attractive; the result is too. A single-factor blend reduces
    to that factor's z-score, so top-N selection is unchanged from the raw factor.

    SURVIVORSHIP: because each panel is standardized cross-sectionally, the inputs must be
    restrict_to_universe()'d to the PIT members in a survivorship-free study, or the union
    leaks into the z-scores (see restrict_to_universe)."""
    weights = weights if weights is not None else [1.0] * len(panels)
    if len(weights) != len(panels):
        raise ValueError("weights must match panels")
    zsum = wsum = None
    for panel, wt in zip(panels, weights):
        z = standardize(panel)
        contrib = (z * wt).fillna(0.0)
        present = z.notna() * float(wt)
        zsum = contrib if zsum is None else zsum.add(contrib, fill_value=0.0)
        wsum = present if wsum is None else wsum.add(present, fill_value=0.0)
    return zsum / wsum.where(wsum > 0)


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
    """Negative log market cap (the small-size premium).

    NOT in the evaluated factor set yet: market-cap ingestion is DEFERRED (Tushare
    daily_basic is rate-limited on the current tier; see scripts/ingest_size.py).
    """
    return -np.log(total_mv.where(total_mv > 0))
