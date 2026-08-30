"""Calendar-anchor schedule with cross-month roll-forward (issue #23).

The blocking property is D=1 parity: the new semantics must reproduce the legacy monthly rule
exactly, or days 2..31 mean nothing. Everything else here pins the roll-forward behaviour that
distinguishes this from issue #21's clamping variant.
"""
import numpy as np
import pandas as pd
import pytest

from hermes.research.backtest.portfolio import signal_portfolio_backtest
from hermes.research.backtest.schedule import (anchor_rollforward_schedule, assert_no_lookahead,
                                               calendar_rebalance_schedule, month_end_schedule,
                                               schedule_audit_rows)


def _bdays(a, b):
    return pd.bdate_range(a, b)


# --- the blocking gate ---------------------------------------------------------------

def test_d1_reproduces_the_legacy_monthly_schedule():
    for a, b in [("2016-08-29", "2026-08-28"), ("2015-01-05", "2025-12-31"),
                 ("2020-02-03", "2021-03-31")]:
        dates = _bdays(a, b)
        assert anchor_rollforward_schedule(dates, 1) == month_end_schedule(dates), (a, b)


def test_d1_parity_end_to_end_through_the_engine():
    rng = np.random.default_rng(23)
    dates = _bdays("2018-01-01", "2020-12-31")
    codes = [f"c{i}" for i in range(6)]
    price = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(dates), len(codes))), 0)),
                         index=dates, columns=codes)
    sig = pd.DataFrame(rng.normal(size=(len(dates), len(codes))), index=dates, columns=codes)
    base = signal_portfolio_backtest(price, sig, 100_000, 3, collect_trades=True)
    d1 = signal_portfolio_backtest(price, sig, 100_000, 3, collect_trades=True,
                                   schedule=anchor_rollforward_schedule(dates, 1))
    pd.testing.assert_series_equal(base.equity, d1.equity)
    assert base.n_rebalances == d1.n_rebalances
    assert base.total_costs == pytest.approx(d1.total_costs)
    assert base.cagr == pytest.approx(d1.cagr)
    assert base.max_drawdown == pytest.approx(d1.max_drawdown)
    assert [(t["date"], t["code"], t["shares"]) for t in base.trades] == \
           [(t["date"], t["code"], t["shares"]) for t in d1.trades]


# --- roll-forward, the point of this variant -----------------------------------------

def test_cross_month_rollforward_is_allowed_unlike_the_clamping_variant():
    """The 31st is a Sunday ending the month: clamping stays on the 29th, roll-forward crosses."""
    dates = pd.DatetimeIndex(["2021-01-04", "2021-01-28", "2021-01-29", "2021-02-01", "2021-02-08"])
    roll = anchor_rollforward_schedule(dates, 31)
    clamp = calendar_rebalance_schedule(dates, 31)
    assert pd.Timestamp("2021-02-01") in [dates[e] for e in roll]     # crossed into February
    assert pd.Timestamp("2021-01-29") in [dates[e] for e in clamp]    # clamped inside January
    assert roll != clamp


def test_weekend_rolls_forward_to_the_next_trading_bar():
    dates = _bdays("2026-08-03", "2026-09-30")
    sch = anchor_rollforward_schedule(dates, 5)                        # 2026-09-05 is a Saturday
    sept = sorted(dates[e] for e in sch if dates[e].month == 9)
    assert sept == [pd.Timestamp("2026-09-07")]


def test_long_holiday_closure_rolls_to_the_first_bar_after_it():
    """A Spring-Festival-shaped gap swallowing the anchor."""
    dates = pd.DatetimeIndex(["2024-02-08", "2024-02-09", "2024-02-19", "2024-02-20",
                              "2024-03-01"])
    sch = anchor_rollforward_schedule(dates, 12)                       # closed 10th..18th
    feb = [dates[e] for e in sch if dates[e].month == 2]
    assert feb == [pd.Timestamp("2024-02-19")]


def test_february_short_month_clamps_the_nominal_anchor():
    non_leap = _bdays("2019-01-01", "2019-04-30")
    rows = schedule_audit_rows(non_leap, 31, anchor_rollforward_schedule(non_leap, 31))
    feb = next(r for r in rows if r["anchor_month"] == "2019-02")
    assert feb["nominal_anchor_date"] == "2019-02-28"
    leap = _bdays("2020-01-01", "2020-04-30")
    rows_l = schedule_audit_rows(leap, 31, anchor_rollforward_schedule(leap, 31))
    feb_l = next(r for r in rows_l if r["anchor_month"] == "2020-02")
    assert feb_l["nominal_anchor_date"] == "2020-02-29"


def test_thirty_day_month_with_d31_resolves_to_the_thirtieth():
    dates = _bdays("2021-04-01", "2021-06-30")
    rows = schedule_audit_rows(dates, 31, anchor_rollforward_schedule(dates, 31))
    apr = next(r for r in rows if r["anchor_month"] == "2021-04")
    assert apr["nominal_anchor_date"] == "2021-04-30"


# --- invariants -----------------------------------------------------------------------

@pytest.mark.parametrize("d", range(1, 32))
def test_no_lookahead_and_one_rebalance_per_bar(d):
    dates = _bdays("2016-08-29", "2026-08-28")
    sch = anchor_rollforward_schedule(dates, d)
    assert_no_lookahead(sch, dates)
    assert all(s < e for e, s in sch.items())


@pytest.mark.parametrize("d", range(1, 32))
def test_schedule_is_monotonic_and_at_most_one_per_anchor_month(d):
    dates = _bdays("2016-08-29", "2026-08-28")
    rows = schedule_audit_rows(dates, d, anchor_rollforward_schedule(dates, d))
    months = [r["anchor_month"] for r in rows]
    assert months == sorted(months), "anchor months must be ordered"
    assert len(months) == len(set(months)), "at most one rebalance per anchor month"
    execs = [r["execution_date"] for r in rows]
    assert execs == sorted(execs) and len(execs) == len(set(execs))


def test_first_bar_anchor_is_skipped_for_want_of_a_signal():
    dates = _bdays("2020-01-01", "2020-03-31")
    assert 0 not in anchor_rollforward_schedule(dates, 1)


def test_anchor_past_the_panel_end_is_dropped_not_clamped():
    """The panel ends mid-month: a later anchor has no bar and must simply not fire."""
    dates = pd.DatetimeIndex(["2026-08-03", "2026-08-10", "2026-08-17"])
    sch = anchor_rollforward_schedule(dates, 28)
    assert not [e for e in sch if dates[e].month == 8]


def test_two_anchor_months_cannot_claim_the_same_bar_twice():
    """A closure spanning a month boundary can push two anchors onto one bar; the first keeps it."""
    dates = pd.DatetimeIndex(["2026-01-28", "2026-03-02", "2026-03-03"])   # all February closed
    sch = anchor_rollforward_schedule(dates, 15)
    assert len(sch) == len(set(sch)), "no bar claimed twice"
    rows = schedule_audit_rows(dates, 15, sch)
    assert len({r["execution_date"] for r in rows}) == len(rows)


def test_out_of_range_day_raises():
    dates = _bdays("2020-01-01", "2020-03-31")
    for bad in (0, 32, -5):
        with pytest.raises(ValueError):
            anchor_rollforward_schedule(dates, bad)


# --- the audit trail ------------------------------------------------------------------

def test_audit_rows_flag_rolls_and_month_crossings():
    dates = pd.DatetimeIndex(["2021-01-04", "2021-01-29", "2021-02-01", "2021-02-26"])
    rows = schedule_audit_rows(dates, 31, anchor_rollforward_schedule(dates, 31))
    jan = next(r for r in rows if r["anchor_month"] == "2021-01")
    assert jan["nominal_anchor_date"] == "2021-01-31"
    assert jan["execution_date"] == "2021-02-01"
    assert jan["rolled_for_weekend_or_holiday"] is True
    assert jan["crossed_calendar_month"] is True


def test_audit_rows_report_no_roll_when_the_anchor_is_a_trading_bar():
    dates = _bdays("2026-06-01", "2026-07-31")
    rows = schedule_audit_rows(dates, 1, anchor_rollforward_schedule(dates, 1))
    jul = next(r for r in rows if r["anchor_month"] == "2026-07")
    assert jul["nominal_anchor_date"] == "2026-07-01" == jul["execution_date"]
    assert jul["rolled_for_weekend_or_holiday"] is False
    assert jul["crossed_calendar_month"] is False


def test_audit_row_count_matches_the_schedule():
    dates = _bdays("2016-08-29", "2026-08-28")
    for d in (1, 15, 31):
        sch = anchor_rollforward_schedule(dates, d)
        assert len(schedule_audit_rows(dates, d, sch)) == len(sch), d
