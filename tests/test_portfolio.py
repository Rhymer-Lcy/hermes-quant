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


def _one_rebalance_then_b_doubles():
    """One rebalance (Jan month-end -> early-Feb exec), then `b` doubles with no further
    rebalance, so end equity reflects exactly the entry weights. `a` stays flat."""
    dates = pd.bdate_range("2020-01-02", "2020-02-14")
    price = pd.DataFrame({"a": 10.0, "b": 10.0}, index=dates)
    price.loc[price.index > pd.Timestamp("2020-02-07"), "b"] = 20.0
    signal = pd.DataFrame({"a": 1.0, "b": 1.0}, index=dates)   # tie -> both held at n_hold=2
    return price, signal


def test_equal_weight_callable_matches_default():
    price, signal = _one_rebalance_then_b_doubles()
    kw = dict(capital=10_000_000, n_hold=2, costs=ZERO_COSTS)
    base = signal_portfolio_backtest(price, signal, **kw)
    explicit = signal_portfolio_backtest(price, signal, weight_asof=lambda d, c: {x: 1.0 for x in c}, **kw)
    # an equal-weight callable must reproduce the default (None) path exactly -> the
    # gross-invested fraction is preserved; weighting only redistributes within the basket.
    assert abs(explicit.total_return - base.total_return) < 1e-9


def test_weighting_shifts_capital_toward_overweighted_name():
    price, signal = _one_rebalance_then_b_doubles()
    kw = dict(capital=10_000_000, n_hold=2, costs=ZERO_COSTS)
    eq = signal_portfolio_backtest(price, signal, **kw)
    fav_b = signal_portfolio_backtest(price, signal, weight_asof=lambda d, c: {"a": 0.2, "b": 0.8}, **kw)
    # `b` doubles; equal weight earns ~+50%, overweighting `b` (0.8) earns ~+80%.
    assert 0.45 < eq.total_return < 0.55
    assert 0.75 < fav_b.total_return < 0.85
    assert fav_b.total_return > eq.total_return
