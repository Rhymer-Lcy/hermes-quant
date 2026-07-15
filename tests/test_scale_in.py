"""Scale-in mechanics: tranche fills, cash drag, delisting truncation, entry discipline."""
import math

import numpy as np
import pandas as pd

from hermes.research.backtest.scale_in import pyramid_vs_lump, run_events

BUY = 0.00076


def test_flat_path_never_deploys_the_lower_tranches():
    r = pyramid_vs_lump(np.array([10.0, 10.0, 10.0]), buy_rate=0.0)
    assert r["filled"] == 1
    assert math.isclose(r["pyramid"], 1.0) and math.isclose(r["lump"], 1.0)


def test_v_shaped_path_fills_cheap_and_beats_lump_sum():
    # Entry 10, crash to 8 (fills -7.5% and -15% at the same close), recover to 10.
    path = np.array([10.0, 8.0, 10.0])
    r = pyramid_vs_lump(path, buy_rate=0.0)
    assert r["filled"] == 3
    assert math.isclose(r["lump"], 1.0)
    assert math.isclose(r["pyramid"], 1.0 / 3 + 2.0 / 3 * (10.0 / 8.0))


def test_straight_rally_leaves_two_thirds_in_cash_and_loses_to_lump_sum():
    path = np.array([10.0, 11.0, 12.0])
    r = pyramid_vs_lump(path, buy_rate=0.0)
    assert r["filled"] == 1
    assert math.isclose(r["lump"], 1.2)
    assert math.isclose(r["pyramid"], 1.2 / 3 + 2.0 / 3)


def test_buy_rate_is_charged_per_filled_tranche_only():
    path = np.array([10.0, 10.0])
    r = pyramid_vs_lump(path, buy_rate=BUY)
    assert math.isclose(r["lump"], 1.0 - BUY)
    assert math.isclose(r["pyramid"], (1.0 - BUY) / 3 + 2.0 / 3)   # cash tranches pay nothing


def test_run_events_enters_the_next_traded_close_and_skips_suspension_straddles():
    dates = pd.bdate_range("2024-01-01", periods=40)
    close = pd.DataFrame({"A": np.linspace(10, 12, 40)}, index=dates)
    close.iloc[6:35, 0] = np.nan                    # a long suspension right after signal two
    events = pd.DataFrame({"code": ["A", "A"], "date": [dates[2], dates[5]]})
    out = run_events(events, close, horizon=3, buy_rate=0.0)
    assert list(out["entry_date"]) == [dates[3]]    # the straddled event is dropped
    assert out["n_days"].iloc[0] == 3


def test_run_events_marks_a_truncated_series_at_its_last_real_close():
    dates = pd.bdate_range("2024-01-01", periods=10)
    close = pd.DataFrame({"A": [10.0] * 5 + [np.nan] * 5}, index=dates)
    events = pd.DataFrame({"code": ["A"], "date": [dates[1]]})
    out = run_events(events, close, horizon=250, buy_rate=0.0)
    assert out["n_days"].iloc[0] == 2               # entry at day 2, series dies at day 4
    assert math.isclose(out["diff"].iloc[0], 0.0)   # flat path -> the arms tie
