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


def test_float_cap_reconstruction_and_zero_turn_guard():
    # float_cap = amount / (turn/100); turn is the free-float turnover rate in PERCENT.
    amount = pd.DataFrame({"a": [1000.0, 1000.0, 1000.0]})
    turn = pd.DataFrame({"a": [2.0, 0.0, -1.0]})       # 2% -> cap 50000; 0 / negative -> NaN
    cap = fl.float_cap(amount, turn)
    assert math.isclose(cap.loc[0, "a"], 50000.0)      # 1000 / (2/100)
    assert np.isnan(cap.loc[1, "a"])                   # turn==0 -> NaN (no spurious inf)
    assert np.isnan(cap.loc[2, "a"])                   # turn<0 -> NaN


def test_small_size_orientation_smaller_is_higher():
    cap = pd.DataFrame({"small": [1e9], "big": [1e12]})
    s = fl.small_size(cap)
    assert s.loc[0, "small"] > s.loc[0, "big"]         # smaller cap = more attractive


def test_roe_quality_orientation_and_nonpositive_guard():
    pe = pd.DataFrame({"hi": [10.0], "lo": [10.0], "loss": [-5.0], "negbook": [10.0]})
    pb = pd.DataFrame({"hi": [3.0], "lo": [1.0], "loss": [2.0], "negbook": [-1.0]})
    r = fl.roe(pe, pb)                                  # ROE = pb/pe where pe>0 & pb>0
    assert math.isclose(r.loc[0, "hi"], 0.3) and math.isclose(r.loc[0, "lo"], 0.1)
    assert r.loc[0, "hi"] > r.loc[0, "lo"]              # higher pb/pe = higher quality
    assert np.isnan(r.loc[0, "loss"]) and np.isnan(r.loc[0, "negbook"])   # loss / negative book -> NaN


def test_trailing_return_computes_ratio():
    close = pd.DataFrame({"a": [10.0, 11.0, 12.0, 15.0]})
    tr = fl.trailing_return(close, lookback=1)
    assert math.isclose(tr.loc[1, "a"], 0.1)           # 11/10 - 1
    assert math.isclose(tr.loc[3, "a"], 0.25)          # 15/12 - 1
    assert np.isnan(tr.loc[0, "a"])                    # no prior bar


def test_restrict_to_universe_empty_date_is_all_nan():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=pd.to_datetime(["2020-01-31", "2020-02-29"]))
    members = {pd.Timestamp("2020-01-31"): {"a", "b"}}   # Feb has NO members

    def asof(d):
        return members.get(pd.Timestamp(d), set())

    out = fl.restrict_to_universe(df, asof)
    assert out.loc["2020-01-31"].notna().all()          # both kept in Jan
    assert out.loc["2020-02-29"].isna().all()           # empty universe -> all NaN


def test_blend_zero_weight_factor_is_ignored():
    f1 = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
    f2 = pd.DataFrame({"a": [3.0], "b": [2.0], "c": [1.0]})   # opposite order
    only_f1 = fl.blend([f1], [1.0])
    with_zero = fl.blend([f1, f2], [1.0, 0.0])               # f2 at weight 0 must not affect
    pd.testing.assert_frame_equal(with_zero, only_f1)
