"""涨跌停 no-fill: limit detection + the engine's buy/sell blocking (opt-in)."""
import numpy as np
import pandas as pd

from hermes.research.backtest.frictions import AShareCosts
from hermes.research.backtest.limits import limit_flags
from hermes.research.backtest.portfolio import execute_orders


def test_limit_flags_main_board_up_down_normal():
    dates = pd.bdate_range("2020-01-01", periods=3)
    preclose = pd.DataFrame({"sh.600000": [10.0, 10.0, 10.0]}, index=dates)
    close = pd.DataFrame({"sh.600000": [11.0, 9.0, 10.5]}, index=dates)   # +10% / -10% / +5%
    f = limit_flags(close, preclose)
    assert f.iloc[0, 0] == 1     # up-limit (main board ±10%)
    assert f.iloc[1, 0] == -1    # down-limit
    assert f.iloc[2, 0] == 0     # normal


def test_limit_flags_chinext_uses_20pct():
    dates = pd.bdate_range("2020-01-01", periods=2)
    preclose = pd.DataFrame({"sz.300001": [10.0, 10.0]}, index=dates)
    close = pd.DataFrame({"sz.300001": [11.0, 12.0]}, index=dates)        # +10% (NOT a limit) / +20% (limit)
    f = limit_flags(close, preclose)
    assert f.iloc[0, 0] == 0     # +10% is below ChiNext's ±20% limit -> tradable
    assert f.iloc[1, 0] == 1     # +20% hits the ChiNext up-limit


def test_limit_flags_nan_where_no_bar():
    dates = pd.bdate_range("2020-01-01", periods=2)
    preclose = pd.DataFrame({"sh.600000": [10.0, np.nan]}, index=dates)
    close = pd.DataFrame({"sh.600000": [11.0, np.nan]}, index=dates)
    assert np.isnan(limit_flags(close, preclose).iloc[1, 0])


def test_execute_orders_blocks_buy_at_up_limit():
    positions: dict[str, int] = {}
    raw = pd.Series({"a": 10.0, "b": 10.0})
    execute_orders(1_000_000.0, positions, {"a": 100, "b": 100}, raw,
                   AShareCosts(), 0.0, 100, block_buy={"a"})
    assert "a" not in positions            # locked limit-up -> buy did not fill
    assert positions.get("b") == 100       # unblocked name fills normally


def test_execute_orders_blocks_sell_at_down_limit():
    positions = {"a": 100}
    raw = pd.Series({"a": 10.0})
    execute_orders(1_000_000.0, positions, {"a": 0}, raw,
                   AShareCosts(), 0.0, 100, block_sell={"a"})
    assert positions.get("a") == 100       # locked limit-down -> sell did not fill (still held)
