"""Backtest engine invariants: the delisting force-liquidation (review bug #1) and a
constant-price zero-cost sanity check."""
import numpy as np
import pandas as pd

from hermes.research.backtest.frictions import ZERO_COSTS
from hermes.research.backtest.portfolio import signal_portfolio_backtest


def test_delisted_holding_is_liquidated_and_capital_recycled():
    dates = pd.bdate_range("2020-01-02", "2020-03-31")
    price = pd.DataFrame(index=dates, dtype=float)
    price["dies"] = 10.0
    price.loc[price.index > pd.Timestamp("2020-02-14"), "dies"] = np.nan   # delists mid-Feb
    price["keep"] = 10.0
    price.loc[price.index > pd.Timestamp("2020-03-02"), "keep"] = 20.0      # doubles once buyable

    signal = pd.DataFrame(index=dates, dtype=float)
    signal["dies"] = 2.0    # preferred while tradable -> bought at the first rebalance
    signal["keep"] = 1.0

    r = signal_portfolio_backtest(price, signal, capital=1_000_000, n_hold=1, costs=ZERO_COSTS)

    # Equity never NaN/phantom; capital was force-liquidated out of the delisted name and
    # recycled into `keep` (which doubled) -> ~+100%. If the delisted holding stayed stuck
    # (the bug), capital would be frozen at a flat price and the return ~0.
    assert r.equity.notna().all()
    assert r.total_return > 0.5


def test_constant_price_zero_cost_is_flat():
    dates = pd.bdate_range("2020-01-02", "2020-04-30")
    price = pd.DataFrame({"a": 10.0, "b": 10.0}, index=dates)
    signal = pd.DataFrame({"a": 1.0, "b": 2.0}, index=dates)
    r = signal_portfolio_backtest(price, signal, capital=1_000_000, n_hold=1, costs=ZERO_COSTS)
    assert abs(r.total_return) < 1e-9
