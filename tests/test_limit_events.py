"""Limit-up event mechanics: streak starts, fillable entries, missed runs, CARs."""
import math

import numpy as np
import pandas as pd

from hermes.research.backtest.limit_events import event_cars, fresh_events, resolve_entries
from hermes.research.backtest.metrics import clustered_tstat, tstat

DATES = pd.bdate_range("2024-01-01", periods=30)


def _panel(values, codes=("A",)):
    return pd.DataFrame({c: values for c in codes}, index=DATES[:len(values)], dtype=float)


def test_a_streak_is_one_event_at_its_start():
    flags = _panel([0, 1, 1, 1, 0, 1, 0])
    ev = fresh_events(flags)
    assert list(ev["date"]) == [DATES[1], DATES[5]]     # two streaks, counted once each


def test_entry_is_the_first_unsealed_close_and_missed_is_the_locked_run():
    flags = _panel([0, 1, 1, 1, 0, 0])
    abn = _panel([0.0, 0.09, 0.10, 0.10, -0.02, 0.0])   # locked days rack up +20% you cannot get
    ev = resolve_entries(fresh_events(flags), flags, abn)
    r = ev.iloc[0]
    assert r["entry_date"] == DATES[4] and r["wait"] == 3
    assert math.isclose(r["missed"], 0.10 + 0.10 - 0.02)   # (event, entry] inclusive of entry day


def test_suspension_does_not_count_as_an_entry():
    flags = _panel([0, 1, np.nan, 0, 0])                 # halted the day after the seal
    abn = _panel([0.0] * 5)
    ev = resolve_entries(fresh_events(flags), flags, abn)
    assert ev["entry_date"].iloc[0] == DATES[3]          # first TRADED unlocked close, not the halt


def test_unresolvable_event_keeps_nat_and_is_dropped_by_cars():
    flags = _panel([0, 1, 1, 1, 1, 1])
    abn = _panel([0.01] * 6)
    ev = resolve_entries(fresh_events(flags), flags, abn, max_wait=3)
    assert pd.isna(ev["entry_date"].iloc[0])
    assert event_cars(ev, abn, horizons=(2,), rt_cost=0.0).empty


def test_car_accrues_from_the_day_after_entry_net_of_cost():
    flags = _panel([0, 1, 0] + [0] * 27)
    abn = _panel([0.0, 0.10, 0.03] + [0.01] * 27)
    ev = resolve_entries(fresh_events(flags), flags, abn)
    out = event_cars(ev, abn, horizons=(5,), rt_cost=0.002)
    r = out.iloc[0]
    assert r["entry_date"] == DATES[2]
    assert math.isclose(r["car_5"], 5 * 0.01 - 0.002)    # entry-day close earns nothing
    assert r["n_days_5"] == 5


def test_horizon_off_the_panel_is_nan_not_truncated():
    flags = _panel([0, 1, 0, 0, 0])
    abn = _panel([0.01] * 5)
    ev = resolve_entries(fresh_events(flags), flags, abn)
    out = event_cars(ev, abn, horizons=(2, 10), rt_cost=0.0)
    assert not math.isnan(out["car_2"].iloc[0])
    assert math.isnan(out["car_10"].iloc[0])


def test_delisted_name_counts_the_days_it_traded():
    flags = _panel([0, 1, 0] + [0] * 27)
    abn = _panel([0.0, 0.0, 0.0, 0.01, 0.01] + [np.nan] * 25)   # dies after day 4
    ev = resolve_entries(fresh_events(flags), flags, abn)
    out = event_cars(ev, abn, horizons=(10,), rt_cost=0.0)
    assert out["n_days_10"].iloc[0] == 2
    assert math.isclose(out["car_10"].iloc[0], 0.02)


def test_down_direction_mirrors():
    flags = _panel([0, -1, -1, 0, 0])
    abn = _panel([0.0, -0.09, -0.10, 0.02, 0.0])
    ev = resolve_entries(fresh_events(flags, direction=-1), flags, abn, direction=-1)
    r = ev.iloc[0]
    assert r["entry_date"] == DATES[3] and r["wait"] == 2
    assert math.isclose(r["missed"], -0.10 + 0.02)


def test_clustered_t_deflates_a_frenzy_month():
    dates = pd.Series([pd.Timestamp("2015-06-05")] * 30
                      + [pd.Timestamp("2015-07-06"), pd.Timestamp("2015-08-05")])
    x = pd.Series([0.05] * 30 + [0.001, -0.001])
    assert tstat(x) > clustered_tstat(x, dates, freq="M")
    assert clustered_tstat(x, dates, freq="M") < 2.0
