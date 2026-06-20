"""Single-name double-MA backtest: entry/exit on the crossover, fees and slippage."""
import numpy as np
import pandas as pd

from hermes.research.backtest.frictions import ZERO_COSTS
from hermes.research.backtest.single_name import double_ma_backtest


def _rise_then_fall():
    dates = pd.bdate_range("2020-01-01", periods=40)
    close = np.concatenate([np.linspace(10, 20, 20), np.linspace(20, 10, 20)])
    return pd.DataFrame({"date": dates, "close": close})


def test_double_ma_enters_long_in_uptrend():
    r = double_ma_backtest(_rise_then_fall(), 1_000_000.0, fast=3, slow=6)
    assert r.n_trades >= 1                              # fast>slow during the rise -> at least one entry


def test_double_ma_exits_after_downturn():
    # A full rise-then-fall produces an entry AND a later exit (>=2 trades), ending flat.
    r = double_ma_backtest(_rise_then_fall(), 1_000_000.0, fast=3, slow=6)
    assert r.n_trades >= 2


def test_double_ma_zero_cost_has_no_costs_but_frictions_do():
    prices = _rise_then_fall()
    free = double_ma_backtest(prices, 1_000_000.0, fast=3, slow=6, costs=ZERO_COSTS)
    paid = double_ma_backtest(prices, 1_000_000.0, fast=3, slow=6)   # default A-share frictions
    assert free.total_costs == 0.0
    assert paid.total_costs > 0.0                      # fees + slippage are charged
    assert paid.equity.iloc[-1] < free.equity.iloc[-1]  # frictions reduce final equity
