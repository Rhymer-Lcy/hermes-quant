"""Rebalance-schedule invariants (issue #21). The D=1 parity gate is the blocking one: the
calendar-timing study is only meaningful if D=1 reproduces the canonical month-end rule exactly,
so it is asserted here on real trading calendars rather than assumed from the derivation."""
import numpy as np
import pandas as pd
import pytest

from hermes.research.backtest.frictions import ZERO_COSTS
from hermes.research.backtest.portfolio import signal_portfolio_backtest
from hermes.research.backtest.schedule import (assert_no_lookahead, calendar_rebalance_schedule,
                                               duplicate_groups, month_end_schedule,
                                               nth_trading_day_schedule, schedule_hash,
                                               turnover_from_trades)


def _trading_days(start, end):
    """Weekday calendar -- enough for schedule logic, which only cares about which bars exist."""
    return pd.bdate_range(start, end)


# --- the blocking gate ---------------------------------------------------------------

def test_d1_reproduces_the_canonical_month_end_schedule():
    for start, end in [("2015-01-05", "2025-12-31"), ("2020-02-03", "2021-03-31"),
                       ("2016-01-01", "2016-12-30")]:
        dates = _trading_days(start, end)
        assert calendar_rebalance_schedule(dates, 1) == month_end_schedule(dates), (start, end)


def test_d1_parity_end_to_end_through_the_engine():
    """Parity must hold on the traded object, not just the index arithmetic: equity, trades and
    metrics from schedule=None and from D=1 must agree."""
    rng = np.random.default_rng(7)
    dates = _trading_days("2018-01-01", "2020-12-31")
    codes = [f"c{i}" for i in range(6)]
    price = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(dates), len(codes))), 0)),
                         index=dates, columns=codes)
    signal = pd.DataFrame(rng.normal(size=(len(dates), len(codes))), index=dates, columns=codes)

    base = signal_portfolio_backtest(price, signal, 1_000_000, 3, collect_trades=True)
    d1 = signal_portfolio_backtest(price, signal, 1_000_000, 3, collect_trades=True,
                                   schedule=calendar_rebalance_schedule(dates, 1))
    pd.testing.assert_series_equal(base.equity, d1.equity)
    assert base.n_rebalances == d1.n_rebalances
    assert base.total_costs == pytest.approx(d1.total_costs)
    assert base.cagr == pytest.approx(d1.cagr) and base.max_drawdown == pytest.approx(d1.max_drawdown)
    assert len(base.trades) == len(d1.trades)
    for a, b in zip(base.trades, d1.trades):
        assert a["date"] == b["date"] and a["code"] == b["code"] and a["shares"] == b["shares"]


def test_schedule_none_is_untouched_by_the_new_argument():
    """The deployed path must not change: omitting `schedule` and passing None agree exactly."""
    rng = np.random.default_rng(3)
    dates = _trading_days("2019-01-01", "2019-12-31")
    price = pd.DataFrame(100 + np.cumsum(rng.normal(0, 1, (len(dates), 4)), 0),
                         index=dates, columns=list("abcd"))
    signal = pd.DataFrame(rng.normal(size=(len(dates), 4)), index=dates, columns=list("abcd"))
    a = signal_portfolio_backtest(price, signal, 500_000, 2, costs=ZERO_COSTS)
    b = signal_portfolio_backtest(price, signal, 500_000, 2, costs=ZERO_COSTS, schedule=None)
    pd.testing.assert_series_equal(a.equity, b.equity)


# --- calendar edge cases -------------------------------------------------------------

def test_february_clamps_and_leap_year():
    dates = _trading_days("2020-01-01", "2020-04-30")          # 2020 is a leap year
    for d in (29, 30, 31):
        sch = calendar_rebalance_schedule(dates, d)
        feb = [dates[e] for e in sch if dates[e].month == 2]
        assert len(feb) == 1
        assert feb[0].day >= 27, feb                            # last trading days of Feb 2020
    non_leap = _trading_days("2019-01-01", "2019-04-30")
    feb19 = [non_leap[e] for e in calendar_rebalance_schedule(non_leap, 31) if non_leap[e].month == 2]
    assert len(feb19) == 1 and feb19[0].day == 28               # 2019-02-28 is a Thursday


def test_thirty_day_month_collapses_d30_and_d31():
    dates = _trading_days("2021-04-01", "2021-06-30")           # April and June have 30 days
    s30, s31 = calendar_rebalance_schedule(dates, 30), calendar_rebalance_schedule(dates, 31)
    apr30 = [dates[e] for e in s30 if dates[e].month == 4]
    apr31 = [dates[e] for e in s31 if dates[e].month == 4]
    assert apr30 == apr31                                       # same target -> same execution bar


def test_target_on_a_weekend_rolls_forward_within_the_month():
    dates = _trading_days("2021-08-01", "2021-08-31")           # 2021-08-01 is a Sunday
    sch = calendar_rebalance_schedule(dates, 1)
    execs = sorted(dates[e] for e in sch)
    assert not execs or execs[0].weekday() < 5


def test_no_trading_day_at_or_after_target_falls_back_to_month_end():
    """A month whose target lands after its last bar must execute on that last bar, not spill into
    the next month -- the pre-registered fallback."""
    dates = pd.DatetimeIndex(["2022-03-01", "2022-03-15", "2022-03-20",       # March ends on the 20th
                              "2022-04-05", "2022-04-25"])
    sch = calendar_rebalance_schedule(dates, 31)
    march = [dates[e] for e in sch if dates[e].month == 3]
    assert march == [pd.Timestamp("2022-03-20")]


def test_long_holiday_gap_rolls_to_the_first_bar_after_it():
    """A Spring-Festival-shaped gap: the target sits inside the closure, so execution is the first
    bar after it, still inside the month."""
    dates = pd.DatetimeIndex(["2023-01-03", "2023-01-19", "2023-01-30", "2023-01-31",
                              "2023-02-01"])                                  # 20th..29th closed
    sch = calendar_rebalance_schedule(dates, 20)
    jan = [dates[e] for e in sch if dates[e].month == 1]
    assert jan == [pd.Timestamp("2023-01-30")]


def test_first_bar_month_is_skipped_for_want_of_a_signal_bar():
    dates = _trading_days("2020-01-01", "2020-03-31")
    sch = calendar_rebalance_schedule(dates, 1)
    assert 0 not in sch                                          # bar 0 has no preceding bar
    assert all(dates[e].month != 1 for e in sch)


def test_one_rebalance_per_month_at_most():
    dates = _trading_days("2015-01-05", "2025-12-31")
    for d in range(1, 32):
        sch = calendar_rebalance_schedule(dates, d)
        months = [(dates[e].year, dates[e].month) for e in sch]
        assert len(months) == len(set(months)), d


# --- invariants and bookkeeping ------------------------------------------------------

@pytest.mark.parametrize("d", range(1, 32))
def test_no_lookahead_for_every_candidate(d):
    dates = _trading_days("2015-01-05", "2025-12-31")
    assert_no_lookahead(calendar_rebalance_schedule(dates, d), dates)


def test_assert_no_lookahead_rejects_a_bad_schedule():
    dates = _trading_days("2020-01-01", "2020-02-28")
    with pytest.raises(AssertionError):
        assert_no_lookahead({5: 5}, dates)                       # signal == exec
    with pytest.raises(AssertionError):
        assert_no_lookahead({5: 6}, dates)                       # signal after exec
    with pytest.raises(AssertionError):
        assert_no_lookahead({9: 5}, dates)                       # stale, non-adjacent


def test_duplicate_detection_groups_identical_schedules():
    dates = _trading_days("2021-04-01", "2021-06-30")
    groups = duplicate_groups({d: calendar_rebalance_schedule(dates, d) for d in range(1, 32)})
    assert sum(len(v) for v in groups.values()) == 31            # every candidate accounted for
    assert any(len(v) > 1 for v in groups.values())              # some genuinely collapse
    for ds in groups.values():
        first = calendar_rebalance_schedule(dates, ds[0])
        assert all(calendar_rebalance_schedule(dates, d) == first for d in ds)


def test_schedule_hash_is_stable_and_discriminating():
    dates = _trading_days("2020-01-01", "2020-12-31")
    a, b = calendar_rebalance_schedule(dates, 1), calendar_rebalance_schedule(dates, 15)
    assert schedule_hash(a) == schedule_hash(dict(a))
    assert schedule_hash(a) != schedule_hash(b)


def test_calendar_day_out_of_range_raises():
    dates = _trading_days("2020-01-01", "2020-03-31")
    for bad in (0, 32, -1):
        with pytest.raises(ValueError):
            calendar_rebalance_schedule(dates, bad)


def test_nth_trading_day_schedule_clamps_and_skips_bar_zero():
    dates = pd.DatetimeIndex(["2022-03-01", "2022-03-02", "2022-04-01", "2022-04-04", "2022-04-05"])
    assert nth_trading_day_schedule(dates, 1) == {2: 1}           # March's 1st bar is bar 0 -> skipped
    assert nth_trading_day_schedule(dates, 2)[1] == 0             # March's 2nd bar
    assert nth_trading_day_schedule(dates, 9)[4] == 3             # clamped to April's last bar
    with pytest.raises(ValueError):
        nth_trading_day_schedule(dates, 0)


def test_turnover_sums_both_sides_at_executed_prices():
    trades = [{"shares": 100, "price": 10.0, "fee": 5.0}, {"shares": -50, "price": 12.0, "fee": 5.0}]
    assert turnover_from_trades(trades) == pytest.approx(100 * 10.0 + 50 * 12.0)


def test_engine_filters_an_out_of_range_or_lookahead_schedule():
    """The engine's own guard: entries violating 0 <= signal < exec < n are dropped, so a bad
    schedule degrades to fewer rebalances rather than silently reading the future."""
    rng = np.random.default_rng(11)
    dates = _trading_days("2020-01-01", "2020-06-30")
    price = pd.DataFrame(100 + np.cumsum(rng.normal(0, 1, (len(dates), 3)), 0),
                         index=dates, columns=list("abc"))
    signal = pd.DataFrame(rng.normal(size=(len(dates), 3)), index=dates, columns=list("abc"))
    bad = {10: 10, 20: 25, 999: 998, 30: 29}                     # only the last is valid
    assert signal_portfolio_backtest(price, signal, 100_000, 2, schedule=bad).n_rebalances == 1
