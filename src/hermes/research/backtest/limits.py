"""Price-limit (daily ±10%/±20% limit) no-fill detection for the friction-faithful backtest.

A name locked at its up-limit has no sellers, so a BUY cannot fill; locked at its down-limit has
no buyers, so a SELL cannot fill. For a close-execution model the question is "is it at the limit
AT THE CLOSE?", detected from the day's close vs the previous close and the board's limit width.

`limit_flags` returns a (date x code) panel: +1 = at/above up-limit (block buys), -1 = at/below
down-limit (block sells), 0 = normal, NaN = no bar. Feed it to signal_portfolio_backtest(
limit_block=...). NOT needed for liquid HS300 large caps (limits rarely bind; cross-checked vs
RQAlpha) and OFF by default there -- this is for the wider CSI500 / small-cap universe (±20%
ChiNext/STAR, frequent limit-locked-at-open days) where limit no-fill materially affects feasibility.

Documented simplifications: uses CURRENT board rules (±10% main, ±20% ChiNext/STAR, ±30% BSE);
ST names' ±5% limit is NOT separately modeled -- filter isST out of the selection universe instead;
pre-2020-08 ChiNext/STAR were ±10% (not modeled -- affects only pre-2020 ChiNext/STAR names).
"""
from __future__ import annotations

import pandas as pd

from ...data.ingest import PRICE_LIMIT_PCT, board_of


def limit_flags(close: pd.DataFrame, preclose: pd.DataFrame, tol_frac: float = 5e-4) -> pd.DataFrame:
    """(date x code) flags: +1 closed at/above the up-limit, -1 at/below the down-limit, else 0.

    `close`/`preclose` are aligned wide panels (forward-adjusted close and previous close, both in the
    daily lake). The limit price is preclose*(1±pct) rounded to 0.01; `tol_frac` absorbs rounding."""
    pct = pd.Series({c: PRICE_LIMIT_PCT.get(board_of(c), 0.10) for c in close.columns})
    up_px = preclose.mul(1.0 + pct, axis=1).round(2)
    dn_px = preclose.mul(1.0 - pct, axis=1).round(2)
    up = close.ge(up_px * (1.0 - tol_frac))
    dn = close.le(dn_px * (1.0 + tol_frac))
    flags = up.astype(int) - dn.astype(int)
    return flags.where(close.notna())
