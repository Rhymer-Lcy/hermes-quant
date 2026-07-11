"""CB shared signal construction: eligibility gates, floors, month-end grid."""
import numpy as np
import pandas as pd

from hermes.cb.signals import (MIN_HISTORY_DAYS, apply_floor, base_and_score,
                               month_end_signals, turnover_metric)


def _panel(days, codes, value):
    idx = pd.bdate_range("2024-01-01", periods=days)
    return pd.DataFrame(value, index=idx, columns=list(codes))


def test_history_gate_needs_prior_days():
    close = _panel(MIN_HISTORY_DAYS + 5, ("110001", "123001"), 100.0)
    close.iloc[:10, 1] = np.nan                       # 123001 starts trading 10 days late
    base, _ = base_and_score(close, close, close * 0.0 + 30.0)
    assert bool(base.iloc[MIN_HISTORY_DAYS, 0])       # 60 prior traded days -> eligible
    assert not bool(base.iloc[MIN_HISTORY_DAYS, 1])   # only 50 -> not yet
    assert not bool(base.iloc[0, 0])                  # day one: no history at all


def test_score_is_em_close_plus_premium_and_gates_on_presence():
    n = MIN_HISTORY_DAYS + 10
    close = _panel(n, ("110001",), 100.0)
    em_close = _panel(n, ("110001",), 120.0)
    prem = _panel(n, ("110001",), 25.0)
    prem.iloc[n - 2] = np.nan                         # premium missing that day
    base, score = base_and_score(close, em_close, prem)
    assert float(score.iloc[0, 0]) == 145.0           # 120 + 25 percentage points
    assert bool(base.iloc[n - 3, 0])                  # history satisfied, inputs present
    assert not bool(base.iloc[n - 2, 0])              # missing premium alone kills the day


def test_floor_is_per_exchange():
    close = _panel(30, ("110001", "123001"), 100.0)
    volume = _panel(30, ("110001", "123001"), 1.0)
    volume["123001"] = 2.0
    base = _panel(30, ("110001", "123001"), True)
    eligible = apply_floor(base, turnover_metric(close, volume),
                           {"11": 150.0, "12": 150.0})
    assert not bool(eligible.iloc[-1, 0])             # SH turnover 100 < its floor
    assert bool(eligible.iloc[-1, 1])                 # SZ turnover 200 >= its floor
    eligible = apply_floor(base, turnover_metric(close, volume),
                           {"11": 50.0, "12": 500.0})
    assert bool(eligible.iloc[-1, 0]) and not bool(eligible.iloc[-1, 1])


def test_turnover_metric_needs_ten_traded_days():
    close = _panel(30, ("110001",), 100.0)
    volume = _panel(30, ("110001",), 10.0)
    close.iloc[10:25] = np.nan                        # long suspension
    m = turnover_metric(close, volume)
    assert np.isnan(m.iloc[26, 0])                    # window holds only 5 traded days
    assert m.iloc[9, 0] == 1000.0                     # min_periods=10 satisfied exactly


def test_month_end_signals_last_trading_day():
    idx = pd.bdate_range("2024-01-01", "2024-03-15")
    ends = month_end_signals(idx, "2024-02-01")
    assert list(ends) == [pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-15")]
