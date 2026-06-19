"""Cross-sectional portfolio backtest with A-share frictions.

Monthly rebalance to an equal-weight top-N basket by a ranking SCORE. Long-only.
No-lookahead: the score is read at the month-end (signal date) and executed at the
NEXT trading day's close. T+1 is respected (monthly holding). 100-share lots,
per-trade 5元 minimum commission, stamp tax, and slippage are all modeled.

Two entry points share one engine (`_score_backtest`):
  - momentum_portfolio_backtest: score = trailing `lookback`-day return.
  - signal_portfolio_backtest:   score = an arbitrary external signal panel
    (e.g. a walk-forward ML model's out-of-sample predictions).

PURPOSE: expose the small-account problem (`avg_names_held` = EFFECTIVE
diversification vs target `n_hold`) and run survivorship-free PIT studies.

Delisting/removal: a holding whose price series permanently ends is force-liquidated
once at its last real price net of fees and is never valued past that bar -- otherwise
a dead name would re-enter P&L at a stale forward-filled price (survivorship bias).

DOCUMENTED SIMPLIFICATIONS (not hidden): (1) 涨跌停 no-fill is not modeled here (rare
for a monthly rebalance on liquid HS300); see the RQAlpha cross-check. (2) Lot-sizing
uses the 前复权 price LEVEL (approximate at the smallest tiers; cross-check confirms it
is second-order). Friction/feasibility tool, not validated alpha.
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


def _score_backtest(price: pd.DataFrame, scores: pd.DataFrame, capital: float,
                    n_hold: int, costs: AShareCosts | None, members_asof) -> PortfolioResult:
    """Engine: each month hold the equal-weight top-`n_hold` names by `scores` (read
    at the month-end signal date, executed next trading day), with A-share frictions."""
    costs = costs or AShareCosts()
    lot = costs.lot_size
    slip = costs.slip

    price = price.sort_index()
    # Per-column last real bar: do NOT value or hold a delisted/terminated name past
    # its final price (an unbounded ffill would carry a dead name forever).
    last_valid = {c: price[c].last_valid_index() for c in price.columns}
    last_price = {c: price[c].loc[lv] for c, lv in last_valid.items() if lv is not None}
    valuation = price.ffill()                  # carry last price through INTERIOR halts...
    for c, lv in last_valid.items():           # ...but never past a name's final bar
        if lv is not None:
            valuation.loc[valuation.index > lv, c] = np.nan
    dates = price.index
    n = len(dates)

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
        di = dates[i]
        raw = price.iloc[i]                    # raw price = tradability + exec price
        val = valuation.iloc[i]

        # Force-exit holdings whose data has permanently ended (delisting / removal +
        # terminal halt): liquidate once at the last real price net of fees.
        for code in list(positions.keys()):
            lv = last_valid.get(code)
            if lv is not None and di > lv:
                qty = positions.pop(code)
                turnover = qty * last_price[code]
                fee = costs.sell_fees(turnover)
                cash += turnover - fee
                total_costs += fee

        if i in rebal_exec:
            sd = dates[rebal_exec[i]]          # signal (month-end) date
            if sd in scores.index:
                f = scores.loc[sd].dropna()
                f = f[raw.reindex(f.index).notna()]            # tradable at exec
                if members_asof is not None:                   # point-in-time universe
                    f = f[f.index.isin(members_asof(sd))]
                top = f.sort_values(ascending=False).head(n_hold).index.tolist()

                if top:
                    equity_now = cash + _hold_value(positions, val)
                    target_val = equity_now / n_hold

                    desired: dict[str, int] = {}
                    for code in top:
                        p = raw.get(code, np.nan)
                        if not np.isnan(p):
                            desired[code] = int(target_val // (p * (1 + slip) * lot)) * lot
                    for code in list(positions.keys()):
                        desired.setdefault(code, 0)

                    # Sells first (free up cash)...
                    for code, tgt in desired.items():
                        cur = positions.get(code, 0)
                        if tgt >= cur:
                            continue
                        p = raw.get(code, np.nan)
                        if np.isnan(p):
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
                        affordable = int(cash // (ep * lot)) * lot
                        qty = min(tgt - cur, max(affordable, 0))
                        if qty <= 0:
                            continue
                        turnover = qty * ep
                        fee = costs.buy_fees(turnover)
                        if turnover + fee > cash:
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


def momentum_portfolio_backtest(panel: pd.DataFrame, capital: float, n_hold: int = 10,
                                lookback: int = 20, costs: AShareCosts | None = None,
                                members_asof=None) -> PortfolioResult:
    """Top-N by trailing `lookback`-day return. `members_asof`: optional
    callable(signal_date)->set[code] for the point-in-time universe."""
    scores = panel / panel.shift(lookback) - 1.0
    return _score_backtest(panel, scores, capital, n_hold, costs, members_asof)


def signal_portfolio_backtest(price: pd.DataFrame, signal: pd.DataFrame, capital: float,
                              n_hold: int = 10, costs: AShareCosts | None = None,
                              members_asof=None) -> PortfolioResult:
    """Top-N by an external `signal` panel (date x code), e.g. walk-forward ML
    out-of-sample predictions. `price` is the 前复权 close panel for exec/valuation."""
    return _score_backtest(price, signal, capital, n_hold, costs, members_asof)
