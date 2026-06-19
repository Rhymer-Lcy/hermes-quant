"""Factor library: orientation, non-positive guards, cross-sectional processing."""
import math

import numpy as np
import pandas as pd

from hermes.research.factors import library as fl


def test_earnings_yield_nonpositive_to_nan():
    pe = pd.DataFrame({"a": [10.0, -5.0, 0.0]})
    ey = fl.earnings_yield(pe)
    assert math.isclose(ey.loc[0, "a"], 0.1)
    assert np.isnan(ey.loc[1, "a"])   # negative PE -> NaN
    assert np.isnan(ey.loc[2, "a"])   # zero PE -> NaN


def test_book_yield_nonpositive_to_nan():
    pb = pd.DataFrame({"a": [2.0, 0.0, -1.0]})
    by = fl.book_yield(pb)
    assert math.isclose(by.loc[0, "a"], 0.5)
    assert np.isnan(by.loc[1, "a"])
    assert np.isnan(by.loc[2, "a"])


def test_zscore_xs_row_mean0_std1():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
    z = fl.zscore_xs(df)
    assert abs(z.iloc[0].mean()) < 1e-9
    assert abs(z.iloc[0].std() - 1.0) < 1e-9       # pandas std is ddof=1


def test_winsorize_xs_clips_to_quantiles():
    df = pd.DataFrame([list(range(100))], dtype=float)
    w = fl.winsorize_xs(df, 0.1, 0.9)
    assert w.iloc[0].min() >= df.iloc[0].quantile(0.1) - 1e-9
    assert w.iloc[0].max() <= df.iloc[0].quantile(0.9) + 1e-9


def test_momentum_skips_recent_window():
    close = pd.DataFrame({"a": [1.0, 2.0, 4.0, 8.0, 16.0]})
    m = fl.momentum(close, lookback=3, skip=1)
    assert math.isclose(m.loc[4, "a"], 3.0)        # close[3]/close[1]-1 = 8/2-1
    assert np.isnan(m.loc[2, "a"])                 # not enough history


def test_low_vol_orientation_calm_beats_wild():
    close = pd.DataFrame({"calm": [100, 101, 102, 103, 104, 105],
                          "wild": [100, 150, 80, 160, 70, 170]}, dtype=float)
    lv = fl.low_vol(close, window=3)
    assert lv.iloc[-1]["calm"] > lv.iloc[-1]["wild"]   # low vol = higher (more attractive)
