"""Single-name long/flat double-MA backtest with A-share frictions.

Purpose: a transparent, fully-inspectable first loop that demonstrates how much
transaction cost eats returns at different capital sizes. This is a MECHANISM /
friction demo, not an alpha claim — a single-name MA crossover has no edge.

No-lookahead & T+1 safe: the signal is computed from day t-1's close and executed
at day t's close, so a position is never entered and exited on bars it could not
have traded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .frictions import AShareCosts


@dataclass
class BTResult:
    equity: pd.Series          # indexed by date
    total_return: float
    cagr: float
    max_drawdown: float
    n_trades: int
    total_costs: float         # cumulative RMB lost to fees + slippage
    end_cash_idle: float       # cash left uninvested at the end (lot-rounding drag)


def _max_drawdown(equity: np.ndarray) -> float:
    roll_max = np.maximum.accumulate(equity)
    return float((equity / roll_max - 1.0).min())


def double_ma_backtest(prices: pd.DataFrame, capital: float, fast: int = 20, slow: int = 60,
                       costs: AShareCosts | None = None) -> BTResult:
    """`prices`: DataFrame with columns ['date', 'close'] (前复权), ascending."""
    costs = costs or AShareCosts()
    df = prices[["date", "close"]].dropna().sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"]).to_numpy()
    close = df["close"].to_numpy(dtype=float)

    ma_f = pd.Series(close).rolling(fast).mean().to_numpy()
    ma_s = pd.Series(close).rolling(slow).mean().to_numpy()
    want_long = ma_f > ma_s  # desired state inferred from each day's close

    slip = costs.slip
    cash = float(capital)
    shares = 0
    total_costs = 0.0
    n_trades = 0
    equity = np.empty(len(close))

    for t in range(len(close)):
        # Execute the PRIOR day's signal at today's close (no lookahead, T+1 safe).
        if t > 0 and not np.isnan(ma_s[t - 1]):
            if want_long[t - 1] and shares == 0:                  # enter long
                exec_price = close[t] * (1 + slip)
                qty = int(cash // (exec_price * costs.lot_size)) * costs.lot_size
                if qty > 0:
                    turnover = qty * exec_price
                    fee = costs.buy_fees(turnover)
                    cash -= turnover + fee
                    total_costs += (turnover - qty * close[t]) + fee  # slippage + fee
                    shares = qty
                    n_trades += 1
            elif not want_long[t - 1] and shares > 0:             # exit to flat
                exec_price = close[t] * (1 - slip)
                turnover = shares * exec_price
                fee = costs.sell_fees(turnover)
                cash += turnover - fee
                total_costs += (shares * close[t] - turnover) + fee
                shares = 0
                n_trades += 1
        equity[t] = cash + shares * close[t]

    years = max((dates[-1] - dates[0]) / np.timedelta64(365, "D"), 1e-9)
    total_return = equity[-1] / capital - 1.0
    cagr = (equity[-1] / capital) ** (1.0 / years) - 1.0
    return BTResult(
        equity=pd.Series(equity, index=pd.to_datetime(df["date"])),
        total_return=float(total_return),
        cagr=float(cagr),
        max_drawdown=_max_drawdown(equity),
        n_trades=n_trades,
        total_costs=float(total_costs),
        end_cash_idle=float(cash if shares == 0 else cash),
    )
