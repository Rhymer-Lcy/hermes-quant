"""Cross-sectional momentum portfolio backtest with A-share frictions.

Monthly rebalance to an equal-weight top-N basket by trailing return. Long-only.
No-lookahead: signal from the month-end close, executed at the NEXT trading day's
close. T+1 is respected (monthly holding). 100-share lots, per-trade 5元 minimum
commission, stamp tax, and slippage are all modeled.

PURPOSE: expose the small-account problem. At 5k–3万 capital the 100-share lot
makes it impossible to actually hold N names, so diversification collapses and the
per-trade minimum commission bites hardest. `avg_names_held` reports the EFFECTIVE
diversification vs the target `n_hold`.

DOCUMENTED SIMPLIFICATION (not hidden): 涨跌停 no-fill is not modeled here (rare for
a monthly rebalance on liquid HS300). It is added in the RQAlpha cross-check step.
This is a friction/feasibility demo, not a validated alpha.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .frictions import AShareCosts


@dataclass
class PortfolioResult:
    equity: pd.Series
    total_return: float
    cagr: float
    max_drawdown: float
    n_rebalances: int
    target_n_hold: int
    avg_names_held: float      # EFFECTIVE diversification (mean held names/day)
    total_costs: float


def _max_drawdown(eq: np.ndarray) -> float:
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def _hold_value(positions: dict[str, int], val: pd.Series) -> float:
    tot = 0.0
    for code, sh in positions.items():
        p = val.get(code, 0.0)
        if not np.isnan(p):
            tot += sh * p
    return tot


def momentum_portfolio_backtest(panel: pd.DataFrame, capital: float, n_hold: int = 10,
                                lookback: int = 20, costs: AShareCosts | None = None) -> PortfolioResult:
    costs = costs or AShareCosts()
    lot = costs.lot_size
    slip = costs.slip

    panel = panel.sort_index()
    valuation = panel.ffill()                 # mark-to-market (carry last price through halts)
    dates = panel.index
    n = len(dates)

    # Month-end signal dates -> execute on the next trading day (no lookahead).
    periods = dates.to_period("M")
    pos_of = {d: i for i, d in enumerate(dates)}
    month_end = pd.Series(dates, index=dates).groupby(periods).max().tolist()
    rebal_exec = {pos_of[sig] + 1: pos_of[sig] for sig in month_end if pos_of[sig] + 1 < n}

    cash = float(capital)
    positions: dict[str, int] = {}
    total_costs = 0.0
    names_held_daily = []
    equity = np.empty(n)

    for i in range(n):
        raw = panel.iloc[i]                   # raw price = tradability + exec price
        val = valuation.iloc[i]               # ffilled = valuation

        if i in rebal_exec:
            si = rebal_exec[i]
            if si - lookback >= 0:
                ret = panel.iloc[si] / panel.iloc[si - lookback] - 1.0
                ret = ret.dropna()
                ret = ret[raw.reindex(ret.index).notna()]          # must be tradable at exec
                top = ret.sort_values(ascending=False).head(n_hold).index.tolist()

                equity_now = cash + _hold_value(positions, val)
                target_val = equity_now / n_hold

                desired: dict[str, int] = {}
                for code in top:
                    p = raw.get(code, np.nan)
                    if np.isnan(p):
                        continue
                    desired[code] = int(target_val // (p * (1 + slip) * lot)) * lot
                for code in list(positions.keys()):
                    desired.setdefault(code, 0)

                # Sells first (free up cash)...
                for code, tgt in desired.items():
                    cur = positions.get(code, 0)
                    if tgt >= cur:
                        continue
                    p = raw.get(code, np.nan)
                    if np.isnan(p):                                # untradable today -> skip
                        continue
                    qty = cur - tgt
                    ep = p * (1 - slip)
                    turnover = qty * ep
                    fee = costs.sell_fees(turnover)
                    cash += turnover - fee
                    total_costs += (qty * p - turnover) + fee
                    positions[code] = cur - qty
                    if positions[code] == 0:
                        del positions[code]

                # ...then buys (capped by available cash).
                for code, tgt in desired.items():
                    cur = positions.get(code, 0)
                    if tgt <= cur:
                        continue
                    p = raw.get(code, np.nan)
                    if np.isnan(p):
                        continue
                    ep = p * (1 + slip)
                    want = tgt - cur
                    affordable = int(cash // (ep * lot)) * lot
                    qty = min(want, max(affordable, 0))
                    if qty <= 0:
                        continue
                    turnover = qty * ep
                    fee = costs.buy_fees(turnover)
                    if turnover + fee > cash:                      # fees pushed over -> drop a lot
                        qty -= lot
                        if qty <= 0:
                            continue
                        turnover = qty * ep
                        fee = costs.buy_fees(turnover)
                    cash -= turnover + fee
                    total_costs += (turnover - qty * p) + fee
                    positions[code] = cur + qty

        equity[i] = cash + _hold_value(positions, val)
        names_held_daily.append(sum(1 for sh in positions.values() if sh > 0))

    years = max((dates[-1] - dates[0]).days / 365.25, 1e-9)
    return PortfolioResult(
        equity=pd.Series(equity, index=dates),
        total_return=float(equity[-1] / capital - 1.0),
        cagr=float((equity[-1] / capital) ** (1.0 / years) - 1.0),
        max_drawdown=_max_drawdown(equity),
        n_rebalances=len(rebal_exec),
        target_n_hold=n_hold,
        avg_names_held=float(np.mean(names_held_daily)),
        total_costs=float(total_costs),
    )
