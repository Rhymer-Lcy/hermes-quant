"""Walk-forward ML: the no-look-ahead invariants (PIT restriction + train-before-predict)."""
import numpy as np
import pandas as pd

from hermes.research.model import walk_forward as wf


def _panel(dates, codes, rng):
    return pd.DataFrame(rng.normal(size=(len(dates), len(codes))), index=dates, columns=codes)


def test_build_dataset_restricts_to_pit_members_and_demeans():
    dates = list(pd.bdate_range("2020-01-31", periods=3, freq="BME"))
    codes = ["a", "b", "c", "d"]
    rng = np.random.default_rng(0)
    close = _panel(dates, codes, rng).abs() + 1.0
    factors = {"f1": _panel(dates, codes, rng)}
    members = {"a", "b", "c"}                          # 'd' is NOT a member

    data, cols = wf.build_dataset(factors, close, dates, members_asof=lambda _t: members)
    assert cols == ["f1"]
    assert "d" not in set(data["code"])                # PIT restriction drops the non-member
    # fwd_ret is cross-sectionally demeaned per date (market/beta removed).
    for _t, g in data.groupby("date"):
        assert abs(g["fwd_ret"].mean()) < 1e-9


def test_walk_forward_predict_never_predicts_before_min_train():
    dates = list(pd.bdate_range("2020-01-31", periods=15, freq="BME"))
    codes = [f"s{i:02d}" for i in range(30)]           # enough names to clear the 200-row train guard
    rng = np.random.default_rng(1)
    close = (_panel(dates, codes, rng).cumsum() + 50.0)
    factors = {"mom": _panel(dates, codes, rng), "vol": _panel(dates, codes, rng)}
    asof = lambda _t: set(codes)

    data, cols = wf.build_dataset(factors, close, dates, members_asof=asof)
    signal = wf.walk_forward_predict(data, cols, min_train=10, window=10)

    assert not signal.empty
    udates = sorted(pd.to_datetime(data["date"].unique()))
    first_allowed = udates[10]                          # min_train -> no prediction before the 11th date
    assert signal.index.min() >= first_allowed
    assert signal.index.max() <= udates[-1]
