"""CB lake cross-checks: close agreement and revision-jump detection."""
import math

import pandas as pd

from hermes.cb.checks import close_mismatch, revision_jump_matched


def _long(code, dates, closes):
    return pd.DataFrame({"code": code, "date": pd.to_datetime(dates), "close": closes})


def test_close_mismatch_zero_when_identical():
    a = _long("A", ["2024-01-02", "2024-01-03"], [100.0, 101.0])
    out = close_mismatch(a, a.copy())
    assert out["rows"] == 2 and out["mismatch_rate"] == 0.0


def test_close_mismatch_counts_deviations():
    sina = _long("A", ["2024-01-02", "2024-01-03"], [100.0, 100.0])
    em = _long("A", ["2024-01-02", "2024-01-03"], [100.0, 101.0])   # 1% off on one row
    out = close_mismatch(sina, em, tol=0.005)
    assert math.isclose(out["mismatch_rate"], 0.5) and math.isclose(out["worst"], 0.01)


def _cv(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_revision_jump_detected():
    cv = _cv([50, 50, 50, 100, 100, 100])            # conversion price 10 -> 5 doubles it
    assert revision_jump_matched(cv, 10.0, 5.0, cv.index[3]) is True


def test_revision_jump_absent():
    cv = _cv([50, 50, 50, 51, 51, 51])
    assert revision_jump_matched(cv, 10.0, 5.0, cv.index[3]) is False


def test_revision_outside_series_unresolved():
    cv = _cv([50, 50, 50])
    late = cv.index[-1] + pd.Timedelta(days=30)
    assert revision_jump_matched(cv, 10.0, 5.0, late) is None
    assert revision_jump_matched(cv, 5.0, 10.0, cv.index[1]) is None   # not a down-revision
