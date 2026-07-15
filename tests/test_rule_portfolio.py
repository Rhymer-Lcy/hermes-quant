"""Rule-portfolio machinery: state persistence, EW rebalance math, fractional targets, boxes."""
import math

import numpy as np
import pandas as pd

from hermes.research.backtest.frictions import ZERO_COSTS, AShareCosts
from hermes.research.backtest.metrics import sharpe
from hermes.research.backtest.rule_portfolio import (box_target, fractional_target_backtest,
                                                     month_end_dates, monthly_ew_backtest,
                                                     proportional_rates, rolling_own_quantile,
                                                     state_mask)

DATES = pd.bdate_range("2024-01-01", periods=60)


def _panel(values, codes=("A",), dates=None):
    dates = DATES if dates is None else dates
    return pd.DataFrame({c: list(values) for c in codes}, index=dates[:len(values)])


def test_proportional_rates_match_the_documented_retail_levels():
    buy, sell = proportional_rates(AShareCosts())
    assert math.isclose(buy, 2.5e-4 + 1e-5 + 5e-4)            # commission + transfer + slippage
    assert math.isclose(sell, buy + 5e-4)                     # plus sell-side stamp tax


def test_state_mask_persists_between_signals_and_exit_wins_a_tie():
    enter = _panel([True, False, False, True, False])
    exit_ = _panel([False, False, True, True, False])
    held = state_mask(enter, exit_)
    assert list(held["A"]) == [True, True, False, False, False]   # day 3: both fire -> exit wins


def test_state_mask_before_any_signal_uses_start_held():
    enter = _panel([False, False, True])
    exit_ = _panel([False, False, False])
    assert list(state_mask(enter, exit_, start_held=True)["A"]) == [True, True, True]
    assert list(state_mask(enter, exit_)["A"]) == [False, False, True]


def test_monthly_ew_backtest_earns_the_held_names_mean_return():
    dates = pd.bdate_range("2024-01-25", periods=25)          # spans a Jan month-end rebalance
    ret = pd.DataFrame({"A": 0.01, "B": 0.03}, index=dates)
    close = (1 + ret).cumprod()
    eligible = close.notna()                                  # both names, always
    res = monthly_ew_backtest(eligible, ret, costs=ZERO_COSTS)
    live = res["gross"][res["gross"] != 0]
    # An EW basket of +1% and +3% names compounds at slightly MORE than 2%/day (the faster
    # name's weight drifts up); day one is exactly the 2% mean, later days exceed it.
    assert math.isclose(live.iloc[0], 0.02, abs_tol=1e-12)
    assert live.iloc[-1] > 0.02
    assert (res["n_held"] == 2).all()


def test_monthly_ew_backtest_charges_buys_and_sells_at_their_own_rates():
    dates = pd.bdate_range("2024-01-25", periods=30)
    ret = pd.DataFrame({"A": 0.0, "B": 0.0}, index=dates)
    # Eligible flips from {A} at the January month-end to {B} at the February month-end.
    eligible = pd.DataFrame({"A": [True] * 30, "B": [False] * 30}, index=dates)
    feb_end = month_end_dates(dates)[1]
    eligible.loc[dates >= feb_end, "A"] = False
    eligible.loc[dates >= feb_end, "B"] = True
    costs = AShareCosts()
    buy, sell = proportional_rates(costs)
    res = monthly_ew_backtest(eligible, ret, costs=costs)
    charged = -res["net"][res["net"] != 0]
    assert math.isclose(charged.iloc[0], buy)                 # initial buy of A
    assert math.isclose(charged.iloc[1], buy + sell)          # sell all of A, buy all of B
    assert math.isclose(res["turnover"].iloc[1], 1.0)         # one-sided turnover = 100%


def test_monthly_ew_backtest_with_nothing_eligible_sits_in_cash():
    dates = pd.bdate_range("2024-01-25", periods=25)
    ret = pd.DataFrame({"A": 0.02}, index=dates)
    eligible = pd.DataFrame({"A": [False] * 25}, index=dates)
    res = monthly_ew_backtest(eligible, ret)
    assert (res["net"] == 0).all() and (res["n_held"] == 0).all()


def test_fractional_target_backtest_scales_return_and_prices_the_trade():
    dates = pd.bdate_range("2024-01-01", periods=4)
    ret = pd.DataFrame({"A": [0.0, 0.10, 0.10, 0.10]}, index=dates)
    target = pd.DataFrame({"A": [1.0, 1.0, 0.5, 0.5]}, index=dates)
    costs = AShareCosts()
    buy, sell = proportional_rates(costs)
    res = fractional_target_backtest(target, ret, costs=costs)
    assert math.isclose(res["gross"].iloc[1, 0], 0.10)        # full position earns the move
    assert math.isclose(res["net"].iloc[2, 0], 0.10 - 0.5 * sell)   # halving pays sell on 0.5
    assert math.isclose(res["gross"].iloc[3, 0], 0.05)        # half position earns half


def test_fractional_target_fraction_drifts_between_signals_instead_of_free_rebalancing():
    dates = pd.bdate_range("2024-01-01", periods=4)
    ret = pd.DataFrame({"A": [0.0, 0.10, 0.10, 0.0]}, index=dates)
    target = pd.DataFrame({"A": [0.5, 0.5, 0.5, 0.5]}, index=dates)
    res = fractional_target_backtest(target, ret, costs=ZERO_COSTS)
    # After a +10% day the held half GREW: fraction 0.5*1.1/1.05, so day 3 earns more than 5%.
    assert math.isclose(res["gross"].iloc[1, 0], 0.05)
    assert math.isclose(res["gross"].iloc[2, 0], 0.5 * 1.1 / 1.05 * 0.10)
    assert (res["traded"].to_numpy() == 0).all()              # constant target -> no trades


def test_box_target_lightens_at_the_top_and_refills_at_the_bottom():
    n = 12
    closes = [10.0] * 5 + [19.5, 15.0, 10.5, 15.0, 19.5, 15.0, 10.0]
    close = _panel(closes, dates=pd.bdate_range("2024-01-01", periods=n))
    # A 5-day box: after the warmup the box spans whatever the prior 5 closes covered.
    tgt = box_target(close, window=5, lo=0.1, hi=0.9, light=0.5)
    assert (tgt["A"].iloc[:5] == 1.0).all()                   # no box yet -> just hold
    assert tgt["A"].iloc[9] == 0.5                            # 19.5 in the top slice -> lighten
    assert tgt["A"].iloc[10] == 0.5                           # mid-box -> previous target persists
    assert tgt["A"].iloc[11] == 1.0                           # bottom slice -> refill


def test_threshold_reversal_state_arms_buys_and_sells_in_order():
    from hermes.research.backtest.rule_portfolio import threshold_reversal_state
    # 5-day lookback, 2-day reversal: dive to a trough, rebound (buy), rally to a peak (sell).
    s = pd.Series([10.0, 9.0, 8.0, 7.0, 6.0, 6.2, 6.4, 6.6, 14.0, 15.0],
                  index=pd.bdate_range("2024-01-01", periods=10))
    st = threshold_reversal_state(s, arm_pct=0.3, sell_pct=0.8, lookback=5, reversal=2)
    assert (st.iloc[:6] == 0).all()             # flat through the dive (momentum still down)
    assert st.iloc[6] == 1 and st.iloc[7] == 1  # armed at the trough, bought on the rebound
    assert st.iloc[8] == 0                      # sold at the peak percentile
    # A rebound WITHOUT a prior trough must not buy: strictly rising series stays flat until
    # armed (it never is), even though its momentum is positive throughout.
    up = pd.Series(np.linspace(10, 20, 10), index=s.index)
    assert (threshold_reversal_state(up, 0.3, 0.8, 5, 2) == 0).all()


def test_rolling_own_quantile_requires_min_obs():
    s = _panel(range(1, 21), dates=pd.bdate_range("2024-01-01", periods=20))
    q = rolling_own_quantile(s.astype(float), q=0.5, window=10, min_obs=5)
    assert q["A"].iloc[:4].isna().all()
    assert math.isclose(q["A"].iloc[4], 3.0)                  # median of 1..5


def test_sharpe_annualizes_and_guards_zero_dispersion():
    up = pd.Series([0.01, 0.02, 0.01, 0.02] * 30)
    assert sharpe(up) > 0
    assert math.isnan(sharpe(pd.Series([0.01] * 100)))        # constant series -> NaN, not 1e16
