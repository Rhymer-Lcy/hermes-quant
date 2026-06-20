"""Index-futures (IF, 沪深300 股指期货) short overlay -- hedge the long HS300 book's systematic
market beta. This is the ONE lever shown able to cut the systematic ~-33% drawdown; selection,
weighting, factor diversification, universe, and cadence are all exhausted (docs/risk_control.md).

Faithful overlay on an existing long-book equity curve:
  - short N INTEGER contracts (CFFEX 沪深300 multiplier = ¥300/index-point), N reset each rebalance
    to round(hedge_ratio * total_equity / (index * mult));
  - daily futures P&L = -N * mult * Δindex flows to a cash sleeve; total = long_book + Σ fut_pnl - costs;
  - LOT GRANULARITY is first-class: 1 IF contract ≈ index*300 ≈ ¥1.4M notional, so a hedge needs an
    account of ~¥1.4M for ONE contract and several ¥M for fine control. SMALL ACCOUNTS CANNOT HEDGE
    with IF (would need 300ETF options -- a separate tool). `effective_ratio` reports the realized
    hedge after integer rounding (can differ a lot from the target at small accounts).

`annual_cost` brackets futures-specific frictions (commission + roll/basis carry) as a drag on the
hedge notional. NOTE: the hedge leg here is the HS300 INDEX return (the beta being removed) -- the
dominant term; real IF carries a basis (post-2015 IF traded at a deep 贴水/discount, a genuine
NEGATIVE carry for a short), so sweep annual_cost (e.g. 0/2/4%/yr) to bracket it. Using actual IF
futures returns (basis/roll embedded) is the documented gold-standard refinement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

IF_MULT = 300   # CFFEX 沪深300 股指期货: ¥300 per index point


def hedge_overlay(long_equity: pd.Series, index_close: pd.Series, hedge_ratio: float, *,
                  mult: int = IF_MULT, annual_cost: float = 0.0,
                  rebal_freq: str = "M") -> tuple[pd.Series, float, float]:
    """Overlay a short IF hedge on `long_equity`, hedging `hedge_ratio` of the book's beta to
    `index_close`. Returns (hedged_equity, mean_abs_contracts, mean_effective_ratio).

    hedge_ratio=0 returns the long book unchanged. The hedge is rebalanced to integer contracts
    at the first trading day of each `rebal_freq` period; futures P&L accrues daily in between."""
    df = pd.concat([long_equity.rename("L"), index_close.rename("idx")], axis=1).dropna().sort_index()
    if hedge_ratio == 0.0 or len(df) == 0:
        return long_equity, 0.0, 0.0
    L = df["L"].to_numpy(dtype=float)
    idx = df["idx"].to_numpy(dtype=float)
    dates = df.index
    n = len(df)

    periods = dates.to_period(rebal_freq)
    is_rebal = np.empty(n, dtype=bool)
    is_rebal[0] = True
    is_rebal[1:] = periods[1:] != periods[:-1]      # first trading day of each new period
    daily_cost = annual_cost / 252.0

    total = np.empty(n)
    total[0] = L[0]
    contracts = 0
    n_log, eff_log = [], []
    for t in range(n):
        if t > 0:
            fut_pnl = -contracts * mult * (idx[t] - idx[t - 1])           # short: gains when index falls
            carry = daily_cost * abs(contracts) * mult * idx[t - 1]       # roll/basis/commission drag
            total[t] = total[t - 1] + (L[t] - L[t - 1]) + fut_pnl - carry
        if is_rebal[t] and total[t] > 0:
            contracts = int(round(hedge_ratio * total[t] / (idx[t] * mult)))
            n_log.append(abs(contracts))
            eff_log.append(contracts * idx[t] * mult / total[t])
    hedged = pd.Series(total, index=dates, name=f"hedged_{hedge_ratio:g}")
    return hedged, float(np.mean(n_log)) if n_log else 0.0, float(np.mean(eff_log)) if eff_log else 0.0


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.to_numpy(dtype=float)
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def cagr(equity: pd.Series) -> float:
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
