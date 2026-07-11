"""CB double-low engine: costs, selection, death/suspension conventions, turnover."""
import math

import numpy as np
import pandas as pd

from hermes.cb.backtest import double_low_backtest

COST = 0.001


def _close(days=10, codes=("A", "B", "C"), price=100.0):
    idx = pd.bdate_range("2024-01-01", periods=days)
    return pd.DataFrame(price, index=idx, columns=list(codes))


def _score(close, rows):
    """rows: {signal_pos: {code: score}}; unmentioned codes are ineligible (NaN)."""
    df = pd.DataFrame(np.nan, index=[close.index[p] for p in rows], columns=close.columns)
    for p, scores in rows.items():
        for c, s in scores.items():
            df.loc[close.index[p], c] = s
    return df


def test_flat_prices_only_entry_cost():
    close = _close()
    score = _score(close, {0: {"A": 1.0, "B": 2.0}})
    r = double_low_backtest(close, score, n_hold=2, cost_per_side=COST)
    assert r.equity.iloc[0] == 1.0                      # signal close, before entry
    assert math.isclose(r.equity.iloc[-1], 1.0 - COST)  # one side of the round trip
    assert math.isclose(r.total_return, -COST)


def test_one_bond_doubling_gives_half():
    close = _close()
    close.loc[close.index[-1], "A"] = 200.0
    score = _score(close, {0: {"A": 1.0, "B": 2.0}})
    r = double_low_backtest(close, score, n_hold=2, cost_per_side=0.0)
    assert math.isclose(r.equity.iloc[-1], 1.5)


def test_selects_lowest_scores():
    close = _close()
    close.loc[close.index[2]:, "A"] = 1.0     # A crashes -- must not be held
    close.loc[close.index[-1], "C"] = 200.0   # C doubles -- must be held
    score = _score(close, {0: {"A": 3.0, "B": 1.0, "C": 2.0}})
    r = double_low_backtest(close, score, n_hold=2, cost_per_side=0.0)
    assert math.isclose(r.equity.iloc[-1], 0.5 + 0.5 * 2.0)


def test_dead_bond_exits_at_last_close():
    close = _close(days=12)
    close.loc[close.index[3], "A"] = 80.0
    close.loc[close.index[4]:, "A"] = np.nan            # A never trades again
    score = _score(close, {0: {"A": 1.0, "B": 2.0}, 6: {"B": 1.0}})
    r = double_low_backtest(close, score, n_hold=2, cost_per_side=0.0)
    assert math.isclose(r.equity.iloc[5], 0.5 * 0.8 + 0.5)   # marked at the last close
    assert math.isclose(r.equity.iloc[-1], 0.9)              # exited at it, all into B


def test_zero_mark_defaults_to_worthless():
    close = _close(days=12)
    close.loc[close.index[3], "A"] = 80.0
    close.loc[close.index[4]:, "A"] = np.nan
    score = _score(close, {0: {"A": 1.0, "B": 2.0}, 6: {"B": 1.0}})
    r = double_low_backtest(close, score, n_hold=2, cost_per_side=0.0,
                            zero_mark=frozenset({"A"}))
    assert math.isclose(r.equity.iloc[5], 0.5)               # dead => zero, B remains
    assert math.isclose(r.equity.iloc[-1], 0.5)


def test_suspended_sale_waits_for_resume():
    close = _close(days=12)
    close.loc[close.index[7], "A"] = np.nan                  # halted on the exec day only
    close.loc[close.index[8]:, "A"] = 120.0
    score = _score(close, {0: {"A": 1.0, "B": 2.0}, 6: {"B": 1.0}})
    r = double_low_backtest(close, score, n_hold=2, cost_per_side=0.0)
    assert math.isclose(r.equity.iloc[-1], 0.5 * 1.2 + 0.5)  # sold at the resume close
    # the deferred slice must not be treated as investable at the rebalance (no leverage)
    assert math.isclose(r.rebalances["oneway_turnover"].iloc[1], 0.0)


def test_unchanged_selection_trades_nothing():
    close = _close()
    score = _score(close, {0: {"A": 1.0, "B": 2.0}, 5: {"A": 1.0, "B": 2.0}})
    r = double_low_backtest(close, score, n_hold=2, cost_per_side=0.0)
    assert r.rebalances["oneway_turnover"].iloc[1] == 0.0
    assert math.isclose(r.equity.iloc[-1], 1.0)


def test_benchmark_holds_all_eligible():
    close = _close()
    close.loc[close.index[-1]] = [110.0, 120.0, 130.0]
    score = _score(close, {0: {"A": 1.0, "B": 2.0, "C": 3.0}})
    r = double_low_backtest(close, score, n_hold=None, cost_per_side=0.0)
    assert math.isclose(r.equity.iloc[-1], (1.1 + 1.2 + 1.3) / 3)


def test_signal_dates_must_be_trading_days():
    close = _close()
    score = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2030-01-01")])
    try:
        double_low_backtest(close, score, n_hold=1, cost_per_side=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an off-calendar signal date")
