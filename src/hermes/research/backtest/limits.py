"""Price-limit (daily +-10%/+-20% limit) no-fill detection for the friction-faithful backtest.

A name locked at its up-limit has no sellers, so a BUY cannot fill; locked at its down-limit has
no buyers, so a SELL cannot fill. For a close-execution model the question is "is it at the limit
AT THE CLOSE?", detected from the day's close vs the previous close and the board's limit width.

`limit_flags` returns a (date x code) panel: +1 = at/above up-limit (block buys), -1 = at/below
down-limit (block sells), 0 = normal, NaN = no bar. Feed it to signal_portfolio_backtest(
limit_block=...). NOT needed for liquid HS300 large caps (limits rarely bind; cross-checked vs
RQAlpha) and OFF by default there -- this is for the wider CSI500 / small-cap universe where
limit no-fill materially affects feasibility, and for limit-up EVENT detection (limit_up_study),
where the rule must hold on every date, not just today.

The limit width is DATE-AWARE (corrected 2026-07-15; it previously applied today's rules to the
whole history): ChiNext (300/301) was +-10% until the registration-system reform took effect on
2020-08-24 and +-20% from that day; STAR (688) has been +-20% since the board opened; the main
board is +-10% throughout; BSE +-30%. Cadence/CSI500 study numbers published before the fix
under-detected pre-2020-08 ChiNext locks (noted in their docs).

Documented simplifications that remain: ST names' +-5% limit is NOT modeled -- filter isST out of
the universe instead; the no-limit days of a fresh listing (first 5 sessions under the
registration system, first day before it) are not modeled -- a >=limit move there can flag as
locked, so event studies must require a minimum trading history, which excludes those days.
"""
from __future__ import annotations

import pandas as pd

from ...data.ingest import PRICE_LIMIT_PCT, board_of

CHINEXT_20PCT_FROM = pd.Timestamp("2020-08-24")   # ChiNext registration-system reform


def limit_pct_panel(index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """(date x code) limit width in force for each name ON each date."""
    base = pd.Series({c: PRICE_LIMIT_PCT.get(board_of(c), 0.10) for c in columns},
                     dtype=float)
    pct = pd.DataFrame([base.to_numpy()] * len(index), index=index, columns=list(columns))
    chinext = [c for c in columns if board_of(c) == "ChiNext"]
    if chinext:
        pct.loc[pct.index < CHINEXT_20PCT_FROM, chinext] = 0.10
    return pct


def limit_flags(close: pd.DataFrame, preclose: pd.DataFrame, tol_frac: float = 5e-4) -> pd.DataFrame:
    """(date x code) flags: +1 closed at/above the up-limit, -1 at/below the down-limit, else 0.

    `close`/`preclose` are aligned wide panels (forward-adjusted close and previous close, both in the
    daily lake). The limit price is preclose*(1+-pct) rounded to 0.01 with the width in force on
    that DATE (see limit_pct_panel); `tol_frac` absorbs rounding."""
    pct = limit_pct_panel(close.index, close.columns)
    up_px = (preclose * (1.0 + pct)).round(2)
    dn_px = (preclose * (1.0 - pct)).round(2)
    up = close.ge(up_px * (1.0 - tol_frac))
    dn = close.le(dn_px * (1.0 + tol_frac))
    flags = up.astype(int) - dn.astype(int)
    return flags.where(close.notna())
