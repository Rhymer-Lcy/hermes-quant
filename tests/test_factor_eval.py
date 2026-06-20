"""Single-factor evaluation gate: rank IC (Spearman, not Pearson) and quantile spread."""
import pandas as pd

from hermes.research.eval.factor_eval import compute_ic, quantile_returns

CODES = ["a", "b", "c", "d", "e"]


def _two_dates():
    return list(pd.bdate_range("2020-01-31", periods=2, freq="BME"))


def test_compute_ic_perfectly_ordered_factor_is_one():
    dates = _two_dates()
    close = pd.DataFrame({c: [1.0, v] for c, v in zip(CODES, [1, 2, 3, 4, 5])}, index=dates)  # fwd ret a<b<c<d<e
    factor = pd.DataFrame({c: [v, v] for c, v in zip(CODES, [10, 20, 30, 40, 50])}, index=dates)  # same order
    res = compute_ic(factor, close, dates)
    assert abs(res.mean_ic - 1.0) < 1e-9


def test_compute_ic_is_rank_based_not_pearson():
    # Factor and forward return share the SAME ORDER but a non-linear relation: Spearman=1,
    # Pearson<1. If compute_ic used Pearson, mean_ic would be < 1.
    dates = _two_dates()
    close = pd.DataFrame({c: [1.0, v] for c, v in zip(CODES, [2, 3, 4, 5, 101])}, index=dates)   # ret 1,2,3,4,100
    factor = pd.DataFrame({c: [v, v] for c, v in zip(CODES, [1, 2, 3, 4, 5])}, index=dates)
    res = compute_ic(factor, close, dates)
    assert abs(res.mean_ic - 1.0) < 1e-9


def test_quantile_returns_monotone_for_ordered_factor():
    dates = _two_dates()
    close = pd.DataFrame({c: [1.0, v] for c, v in zip(CODES, [1, 2, 3, 4, 5])}, index=dates)
    factor = pd.DataFrame({c: [v, v] for c, v in zip(CODES, [1, 2, 3, 4, 5])}, index=dates)
    q = quantile_returns(factor, close, dates, n_q=5)
    assert q.iloc[-1] > q.iloc[0]                        # top factor quantile out-returns the bottom
    assert list(q.values) == sorted(q.values)           # monotone increasing
