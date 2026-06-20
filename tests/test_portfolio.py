"""Backtest engine invariants: the delisting force-liquidation (review bug #1) and a
constant-price zero-cost sanity check."""
import numpy as np
import pandas as pd

from hermes.research.backtest.frictions import ZERO_COSTS
from hermes.research.backtest.portfolio import (_hold_value, _select_top,
                                                signal_portfolio_backtest, valuation_panel)


def test_valuation_panel_ffills_suspension_but_stops_after_delisting():
    dates = pd.bdate_range("2020-01-01", periods=6)
    price = pd.DataFrame(index=dates, dtype=float)
    price["halt"] = [10.0, np.nan, np.nan, 13.0, 14.0, 15.0]   # interior suspension, resumes
    price["dead"] = [20.0, 21.0, 22.0, np.nan, np.nan, np.nan]  # delists after bar 2
    val, last_valid, last_price = valuation_panel(price)
    # interior gap is forward-filled (carry last price through a halt)...
    assert val.loc[dates[1], "halt"] == 10.0 and val.loc[dates[2], "halt"] == 10.0
    # ...but a name is never valued past its final real bar.
    assert np.isnan(val.loc[dates[3], "dead"]) and np.isnan(val.loc[dates[5], "dead"])
    assert last_valid["dead"] == dates[2] and last_price["dead"] == 22.0


def test_hold_value_ignores_nan_prices():
    val = pd.Series({"a": 10.0, "b": np.nan})
    assert _hold_value({"a": 100, "b": 200}, val) == 1000.0   # b (NaN price) contributes 0


def test_rebalance_freq_changes_cadence_and_defaults_to_monthly():
    dates = pd.bdate_range("2020-01-01", "2020-12-31")
    price = pd.DataFrame({"a": np.linspace(10, 14, len(dates)),
                          "b": np.linspace(20, 18, len(dates))}, index=dates)
    signal = pd.DataFrame({"a": 2.0, "b": 1.0}, index=dates)
    n = {f: signal_portfolio_backtest(price, signal, 1_000_000.0, 1, rebalance_freq=f).n_rebalances
         for f in ("Q", "M", "W")}
    assert n["Q"] < n["M"] < n["W"]                 # quarterly rebalances least, weekly most
    default = signal_portfolio_backtest(price, signal, 1_000_000.0, 1).n_rebalances
    assert default == n["M"]                        # default cadence is monthly


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


def test_select_top_band_zero_is_plain_topn():
    assert _select_top(["a", "b", "c", "d", "e"], {"x", "y"}, 3, 0) == ["a", "b", "c"]


def test_select_top_keeps_incumbent_in_buffer_zone():
    # ranked a>b>c>d>e, n_hold=3, band=2 (exit zone = top5). `e` is held and still in the
    # exit zone (rank 5) though outside the strict top-3 -> kept; slots filled from top-3.
    out = _select_top(["a", "b", "c", "d", "e"], {"e"}, 3, 2)
    assert out[0] == "e" and len(out) == 3 and set(out) == {"e", "a", "b"}


def test_select_top_new_name_must_rank_in_strict_topn():
    # `d` (rank 4, not held) must NOT enter at band=2; with no incumbents -> plain top-3.
    assert _select_top(["a", "b", "c", "d", "e"], set(), 3, 2) == ["a", "b", "c"]


def test_rebalance_buffer_cuts_turnover_cost():
    # `a` always ranks 1st; `b`/`c` alternate the 2nd slot each month. With n_hold=2 and no
    # buffer the 2nd holding churns b<->c every month (cost); a band=1 buffer (exit zone =
    # top-3 = all) keeps the incumbent, so it trades once and then holds -> strictly cheaper.
    dates = pd.bdate_range("2020-01-02", "2020-06-30")
    price = pd.DataFrame(10.0, index=dates, columns=["a", "b", "c"])   # flat prices
    periods = dates.to_period("M")
    uniq = list(dict.fromkeys(periods))
    even = {p for i, p in enumerate(uniq) if i % 2 == 0}
    signal = pd.DataFrame(index=dates)
    signal["a"] = 3.0
    signal["b"] = [2.0 if p in even else 1.0 for p in periods]
    signal["c"] = [1.0 if p in even else 2.0 for p in periods]

    kw = dict(capital=1_000_000, n_hold=2)
    no_buf = signal_portfolio_backtest(price, signal, rebalance_band=0, **kw)
    buf = signal_portfolio_backtest(price, signal, rebalance_band=1, **kw)
    assert buf.total_costs < no_buf.total_costs
