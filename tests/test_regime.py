"""Market-regime exposure: index-trend gate (>=MA -> 1) and as-of carry-forward."""
import numpy as np
import pandas as pd

from hermes.research.backtest import regime


def test_trend_exposure_above_below_ma_and_warmup():
    idx = pd.bdate_range("2020-01-01", periods=6)
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 5.0, 4.0], index=idx)
    exp = regime.trend_exposure(close, ma_window=3)
    assert np.isnan(exp.iloc[0]) and np.isnan(exp.iloc[1])   # MA warm-up -> NaN
    assert exp.iloc[2] == 1.0 and exp.iloc[3] == 1.0          # close >= 3-day MA
    assert exp.iloc[4] == 0.0 and exp.iloc[5] == 0.0          # close < MA after the drop


def test_exposure_lookup_asof_and_default():
    idx = pd.bdate_range("2020-01-01", periods=4)
    exp = pd.Series([np.nan, 1.0, 0.0, 1.0], index=idx)
    asof = regime.exposure_lookup(exp)
    assert asof("2019-12-31") == 1.0                  # before any data -> default fully invested
    assert asof(idx[1]) == 1.0
    assert asof(idx[2]) == 0.0                        # picks the as-of value
    assert asof("2020-01-04") == 0.0                  # Saturday -> carries forward Fri (idx[2]=0.0)
    assert asof(idx[3]) == 1.0
