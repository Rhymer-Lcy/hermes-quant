"""Drawdown-ladder mechanics: episode arming, tranche fills, cost recovery, credit effect."""
import math

import numpy as np
import pandas as pd

from hermes.research.backtest.drawdown_ladder import episodes, ladder_outcome, schedule_into


def _series(values, start="2020-01-01"):
    return pd.Series(list(values), index=pd.bdate_range(start, periods=len(values)),
                     dtype=float)


def test_episode_triggers_once_and_rearms_only_near_the_peak():
    # Peak 100 -> dive to 79 (trigger) -> partial bounce to 90 (no re-arm) -> dive to 75
    # (still same episode) -> recover to 96 (re-arms against the NEW rolling peak of 96) ->
    # dive to 76, which is below 80% of 96 (second trigger).
    s = _series([100, 90, 79, 90, 75, 96, 90, 76])
    ep = episodes(s, dd=0.2, peak_window=5, rearm=0.95)
    assert list(ep["iloc"]) == [2, 7]
    assert math.isclose(ep["peak"].iloc[1], 96.0)             # measured against the new peak


def test_ladder_fills_deepen_and_recovery_is_cost_not_peak():
    # Peak 100; trigger at 80, second tranche at 70, third never; V back to 76.
    s = _series([80, 70, 72, 76, 76])
    out = ladder_outcome(s, trigger_iloc=0, peak=100.0, horizon=4)
    assert out["n_fills"] == 2
    # Cost basis: 1/3 @80 + 1/3 @70 + 1/3 cash -> at 76: 76/80/3 + 76/70/3 + 1/3 > 1.
    assert out["recovered"] and out["recovered_day"] == 3
    assert out["terminal"] > 1.0 > out["lump_terminal"]       # lump @80 ends at 76/80


def test_ladder_never_recovers_on_a_monotone_decline():
    s = _series(np.linspace(80, 40, 30))
    out = ladder_outcome(s, trigger_iloc=0, peak=100.0, horizon=29)
    assert out["n_fills"] == 3 and not out["recovered"]
    assert out["terminal"] < 1.0


def test_dividend_credit_can_rescue_a_shallow_underwater_path():
    # One tranche at 80, then flat at 77 (no deeper fills): the account sits 1.25% underwater
    # forever on the price index. The +2%/yr credit compounds only on the INVESTED third, so
    # recovery needs (77/80)*(1.02)^(t/250) >= 1 -- about 483 trading days.
    s = _series([80.0] + [77.0] * 599)
    flat = ladder_outcome(s, trigger_iloc=0, peak=100.0, horizon=599, div_credit=0.0)
    credited = ladder_outcome(s, trigger_iloc=0, peak=100.0, horizon=599, div_credit=0.02)
    assert flat["n_fills"] == 1 and not flat["recovered"]
    assert credited["recovered"] and 450 < credited["recovered_day"] < 520


def test_schedule_into_marks_the_same_dates_on_the_benchmark():
    idx = pd.bdate_range("2020-01-01", periods=10)
    bench = pd.Series(np.linspace(100, 109, 10), index=idx)
    fills = [idx[0], idx[2], None]                            # third tranche stays cash
    v = schedule_into(bench, fills, idx[-1])
    expect = (109 / 100) / 3 + (109 / 102) / 3 + 1 / 3
    assert math.isclose(v, expect)
