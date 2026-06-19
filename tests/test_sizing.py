"""Inverse-vol weighting and the factor blend: weights favour calmer names, the
per-name cap binds, and a single-factor blend is a no-op on selection."""
import numpy as np
import pandas as pd

from hermes.research.backtest.sizing import _apply_cap, inverse_vol_weighter
from hermes.research.factors import library as fl


def _two_name_close(seed_a=0.005, seed_b=0.05):
    """A calm name `a` and a volatile name `b` over ~1y of business days."""
    dates = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(0)
    a = 10.0 * np.cumprod(1 + rng.normal(0, seed_a, len(dates)))
    b = 10.0 * np.cumprod(1 + rng.normal(0, seed_b, len(dates)))
    return pd.DataFrame({"a": a, "b": b}, index=dates)


def test_inverse_vol_favours_the_calmer_name():
    close = _two_name_close()
    w = inverse_vol_weighter(close, lookback=60, cap=None)(close.index[-1], ["a", "b"])
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["a"] > w["b"]          # the calm name gets the larger weight
    assert w["a"] > 0.8             # ~10x lower vol -> dominant weight when uncapped


def test_equal_vol_gives_equal_weight():
    dates = pd.bdate_range("2020-01-01", periods=120)
    # identical return paths -> identical vol -> equal weights
    rng = np.random.default_rng(1)
    path = 10.0 * np.cumprod(1 + rng.normal(0, 0.02, len(dates)))
    close = pd.DataFrame({"a": path, "b": path.copy()}, index=dates)
    w = inverse_vol_weighter(close, lookback=60)(dates[-1], ["a", "b"])
    assert abs(w["a"] - w["b"]) < 1e-9


def test_weight_cap_binds_and_renormalizes():
    w = _apply_cap(pd.Series({"a": 0.9, "b": 0.07, "c": 0.03}), cap=0.5)
    assert abs(w.sum() - 1.0) < 1e-9
    assert w["a"] <= 0.5 + 1e-9                      # capped
    assert w["b"] > 0.07 and w["c"] > 0.03           # excess spilled pro-rata onto the rest


def test_thin_history_name_uses_median_not_dropped():
    close = _two_name_close()
    close["c"] = np.nan
    close.iloc[-5:, close.columns.get_loc("c")] = [10, 10.1, 10.2, 10.1, 10.3]  # tiny history
    w = inverse_vol_weighter(close, lookback=60, cap=None)(close.index[-1], ["a", "b", "c"])
    assert "c" in w and w["c"] > 0                   # sized via median sigma, not dropped
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_blend_single_factor_is_just_its_zscore_ranking():
    dates = pd.bdate_range("2020-01-01", periods=3)
    # distinct values across names each date (no degenerate zero-variance row)
    f = pd.DataFrame({"a": [1.0, 1, 1], "b": [2.0, 2, 2], "c": [3.0, 3, 3]}, index=dates)
    blended = fl.blend([f])
    # blend of one factor preserves the cross-sectional ordering exactly
    for d in dates:
        assert (blended.loc[d].rank() == f.loc[d].rank()).all()


def test_restrict_to_universe_masks_non_members_per_date():
    dates = pd.bdate_range("2020-01-01", periods=2)
    panel = pd.DataFrame({"a": [1.0, 1.0], "b": [2.0, 2.0], "c": [3.0, 3.0]}, index=dates)
    members = {dates[0]: {"a", "b"}, dates[1]: {"b", "c"}}
    out = fl.restrict_to_universe(panel, lambda d: members[d])
    assert pd.isna(out.loc[dates[0], "c"]) and out.loc[dates[0], "a"] == 1.0   # c not a member t0
    assert pd.isna(out.loc[dates[1], "a"]) and out.loc[dates[1], "c"] == 3.0   # a not a member t1


def test_standardization_reference_set_leaks_without_restriction():
    """The survivorship trap, in miniature: a member's z-score changes when a non-member
    (a later index joiner) sits in the cross-section. restrict_to_universe removes it."""
    d = pd.bdate_range("2020-01-01", periods=1)
    full = pd.DataFrame({"a": [1.0], "b": [2.0], "joiner": [10.0]}, index=d)   # joiner ∈ union only
    members = lambda _d: {"a", "b"}                                            # noqa: E731
    z_leaky = fl.standardize(full)                                  # over {a,b,joiner}
    z_pit = fl.standardize(fl.restrict_to_universe(full, members))  # over {a,b}
    # the outlier joiner distorts a & b's standardized scores; restricting fixes it
    assert abs(z_leaky.loc[d[0], "a"] - z_pit.loc[d[0], "a"]) > 1e-6


def test_blend_averages_two_standardized_factors():
    dates = pd.bdate_range("2020-01-01", periods=1)
    # value likes `a`, low-vol likes `c`; the blend should rank the consensus middle name
    val = pd.DataFrame({"a": [3.0], "b": [2.0], "c": [1.0]}, index=dates)
    lvol = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]}, index=dates)
    blended = fl.blend([val, lvol])
    # symmetric, opposing factors -> all tie near 0; b (middle in both) is the median
    assert abs(blended.loc[dates[0]].std()) < 1e-9
