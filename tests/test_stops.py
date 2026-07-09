"""Per-name stop-loss / take-profit (A9): trigger logic, cost basis, and the no-stop parity gate."""
import numpy as np
import pandas as pd
import pytest

from hermes.research.backtest.frictions import ZERO_COSTS
from hermes.research.backtest.portfolio import signal_portfolio_backtest
from hermes.research.backtest.stops import (StopSpec, exit_fill_price,
                                            update_cost_basis)


# --- StopSpec validation -------------------------------------------------------------

def test_spec_rejects_bad_trigger_and_fractions():
    with pytest.raises(ValueError):
        StopSpec(stop_loss=0.1, trigger="intrabar")
    with pytest.raises(ValueError):
        StopSpec(stop_loss=10.0)          # a fraction, not a percent
    with pytest.raises(ValueError):
        StopSpec(take_profit=0.0)
    assert not StopSpec().active
    assert StopSpec(stop_loss=0.1).active


# --- close-trigger mode --------------------------------------------------------------

def test_close_trigger_exits_at_the_close_on_breach():
    s = StopSpec(stop_loss=0.10, take_profit=0.20)
    assert exit_fill_price(s, 100.0, 89.0) == 89.0      # below the -10% stop -> fill at close
    assert exit_fill_price(s, 100.0, 90.0) == 90.0      # exactly at the stop -> exits
    assert exit_fill_price(s, 100.0, 91.0) is None      # inside the band -> hold
    assert exit_fill_price(s, 100.0, 121.0) == 121.0    # above the +20% take -> fill at close


def test_close_trigger_ignores_intraday_extremes():
    s = StopSpec(stop_loss=0.10, trigger="close")
    # pierced -10% intraday but closed inside the band: close mode does not exit
    assert exit_fill_price(s, 100.0, 95.0, low=80.0, high=96.0) is None


def test_no_exit_when_untradable_or_basis_unknown():
    s = StopSpec(stop_loss=0.10)
    assert exit_fill_price(s, 100.0, np.nan) is None    # cannot sell what cannot be priced
    assert exit_fill_price(s, 0.0, 50.0) is None        # no basis -> no reference
    assert exit_fill_price(StopSpec(), 100.0, 1.0) is None   # overlay disabled


# --- intraday-trigger mode: fills are pessimistic on both sides ----------------------

def test_intraday_stop_fills_at_trigger_when_it_recovers_into_the_close():
    s = StopSpec(stop_loss=0.10, trigger="intraday")
    # touched 88 (below the 90 stop) then closed at 95: a market stop filled at ~90, not 95
    assert exit_fill_price(s, 100.0, 95.0, low=88.0, high=96.0) == 90.0


def test_intraday_stop_fills_below_trigger_on_a_gap_through():
    s = StopSpec(stop_loss=0.10, trigger="intraday")
    # gapped straight to 80 and closed there: the fill is the close, NOT the untouched 90 trigger
    assert exit_fill_price(s, 100.0, 80.0, low=80.0, high=82.0) == 80.0


def test_intraday_take_profit_fills_at_the_limit_never_above():
    s = StopSpec(take_profit=0.20, trigger="intraday")
    # gapped up and closed at 140; a sell limit at 120 fills at 120, not at the better close
    assert exit_fill_price(s, 100.0, 140.0, low=130.0, high=145.0) == 120.0


def test_intraday_stop_wins_when_both_are_touched_on_one_bar():
    s = StopSpec(stop_loss=0.10, take_profit=0.20, trigger="intraday")
    # the bar touched both 88 and 125; the daily bar cannot order them, so the STOP is assumed
    assert exit_fill_price(s, 100.0, 105.0, low=88.0, high=125.0) == 90.0


def test_intraday_without_high_low_never_exits():
    s = StopSpec(stop_loss=0.10, take_profit=0.20, trigger="intraday")
    assert exit_fill_price(s, 100.0, 50.0) is None      # low/high default to NaN


# --- cost basis ----------------------------------------------------------------------

def test_basis_of_a_fresh_buy_is_its_execution_price():
    basis, positions = {}, {"a": 100}
    update_cost_basis(basis, positions, [{"code": "a", "shares": 100, "price": 12.0}])
    assert basis["a"] == pytest.approx(12.0)


def test_basis_is_share_weighted_across_buys():
    # 200 shares already held at 10, buy 100 more at 13 -> (10*200 + 13*100)/300 = 11.0
    basis, positions = {"a": 10.0}, {"a": 300}
    update_cost_basis(basis, positions, [{"code": "a", "shares": 100, "price": 13.0}])
    assert basis["a"] == pytest.approx(11.0)


def test_partial_sell_leaves_basis_and_full_exit_drops_it():
    basis, positions = {"a": 10.0}, {"a": 100}
    update_cost_basis(basis, positions, [{"code": "a", "shares": -200, "price": 20.0}])
    assert basis["a"] == 10.0                 # realised P&L does not re-price remaining shares
    basis, positions = {"a": 10.0}, {}
    update_cost_basis(basis, positions, [{"code": "a", "shares": -300, "price": 20.0}])
    assert "a" not in basis


# --- engine integration --------------------------------------------------------------

def _toy(prices_a: list[float]):
    """One-name universe so the top-1 basket is forced; monthly bars so a rebalance is rare."""
    dates = pd.bdate_range("2020-01-01", periods=len(prices_a))
    price = pd.DataFrame({"sh.600000": prices_a}, index=dates)
    signal = pd.DataFrame({"sh.600000": [1.0] * len(prices_a)}, index=dates)
    return price, signal


def test_no_stops_path_is_bit_identical_to_the_deployed_engine():
    """The deployed book must be untouched: stops=None and an inactive StopSpec both reproduce
    the pre-stops equity curve exactly."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=400)
    codes = [f"sh.60000{i}" for i in range(5)]
    price = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (400, 5)), axis=0)),
                         index=dates, columns=codes)
    signal = pd.DataFrame(rng.normal(size=(400, 5)), index=dates, columns=codes)

    base = signal_portfolio_backtest(price, signal, 1_000_000, n_hold=3)
    none_spec = signal_portfolio_backtest(price, signal, 1_000_000, n_hold=3, stops=None)
    inactive = signal_portfolio_backtest(price, signal, 1_000_000, n_hold=3, stops=StopSpec())

    pd.testing.assert_series_equal(base.equity, none_spec.equity)
    pd.testing.assert_series_equal(base.equity, inactive.equity)
    assert base.total_costs == inactive.total_costs


def test_stop_loss_liquidates_and_the_book_stops_tracking_the_decline():
    # rises, then collapses; a -10% stop must exit and hold cash through the rest of the fall
    prices = [10.0] * 3 + [9.5, 8.9, 7.0, 5.0, 4.0] + [4.0] * 5
    price, signal = _toy(prices)
    stopped = signal_portfolio_backtest(price, signal, 100_000, n_hold=1, costs=ZERO_COSTS,
                                        stops=StopSpec(stop_loss=0.10), initial_rebalance=True)
    held = signal_portfolio_backtest(price, signal, 100_000, n_hold=1, costs=ZERO_COSTS,
                                     initial_rebalance=True)
    assert stopped.equity.iloc[-1] > held.equity.iloc[-1]     # exiting beat riding it down
    assert stopped.max_drawdown > held.max_drawdown           # (both negative; stopped is shallower)


def test_a_name_is_never_stopped_on_the_bar_that_bought_it():
    """Entry bar cannot breach: exits are evaluated before the rebalance that establishes basis."""
    price, signal = _toy([10.0, 10.0, 10.0])
    r = signal_portfolio_backtest(price, signal, 100_000, n_hold=1, costs=ZERO_COSTS,
                                  stops=StopSpec(stop_loss=0.001), initial_rebalance=True,
                                  collect_trades=True)
    buys = [t for t in r.trades if t["shares"] > 0]
    assert len(buys) == 1                                     # bought once
    # the basis includes slippage, but ZERO_COSTS has none, so a flat price never breaches
    assert not [t for t in r.trades if t["shares"] < 0]


def test_take_profit_exits_into_strength():
    price, signal = _toy([10.0, 10.0, 12.5, 12.5, 12.5])
    r = signal_portfolio_backtest(price, signal, 100_000, n_hold=1, costs=ZERO_COSTS,
                                  stops=StopSpec(take_profit=0.20), initial_rebalance=True,
                                  collect_trades=True)
    sells = [t for t in r.trades if t["shares"] < 0]
    assert sells and sells[0]["price"] == pytest.approx(12.5)   # exited at the breaching close
