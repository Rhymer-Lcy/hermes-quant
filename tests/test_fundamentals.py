"""Fundamentals lake logic: CSRC bucket mapping and the publication-date-aligned quality gate."""
import pandas as pd

from hermes.data.fundamentals import bucket_of, csrc_class, quality_mask

DATES = pd.bdate_range("2015-01-01", periods=10)


def test_csrc_class_reads_the_ascii_prefix_and_rejects_untagged_names():
    assert csrc_class("J66货币金融服务") == "J66"
    assert csrc_class("") is None and csrc_class(None) is None


def test_bucket_of_follows_the_frozen_five_bucket_map():
    assert bucket_of("J66货币金融服务") == "finance"
    assert bucket_of("C15酒、饮料和精制茶制造业") == "consumption"
    assert bucket_of("I65软件和信息技术服务业") == "tech"
    assert bucket_of("A01农业") is None                     # outside the map -> no bucket


def _roe(code, pairs):
    return pd.DataFrame({"code": code,
                         "pubDate": pd.to_datetime([p for p, _ in pairs]),
                         "statDate": pd.to_datetime([f"{2011 + i}-12-31"
                                                     for i in range(len(pairs))]),
                         "roeAvg": [r for _, r in pairs]})


def test_quality_gate_counts_a_report_only_from_its_publication_date():
    # Three straight >15% years, but the third report publishes mid-window: the gate must
    # flip exactly on the publication day, not on the fiscal-year-end date.
    pub3 = DATES[6]
    roe = _roe("A", [("2013-04-01", 0.20), ("2014-04-01", 0.18), (pub3, 0.22)])
    mask = quality_mask(DATES, ["A"], roe=roe)
    assert not mask["A"].iloc[5]
    assert mask["A"].loc[pub3:].all()


def test_quality_gate_needs_all_recent_reports_above_threshold():
    roe = _roe("A", [("2013-04-01", 0.20), ("2014-04-01", 0.10), ("2014-12-01", 0.22)])
    mask = quality_mask(DATES, ["A"], roe=roe)
    assert not mask["A"].any()                              # the 10% year poisons the window


def test_quality_gate_requires_three_published_reports():
    roe = _roe("A", [("2013-04-01", 0.20), ("2014-04-01", 0.18)])
    assert not quality_mask(DATES, ["A"], roe=roe)["A"].any()


def test_trailing_yield_sees_ex_dates_before_the_price_window_and_ages_them_out():
    from hermes.data.fundamentals import trailing_yield
    idx = pd.bdate_range("2015-01-05", periods=300)
    raw = pd.DataFrame({"A": 10.0}, index=idx)
    div = pd.DataFrame({"code": ["A", "A"],
                        "ex_date": pd.to_datetime(["2014-07-01", "2015-06-01"]),
                        "dps": [0.5, 0.6]})
    y = trailing_yield(div, raw)["A"]
    assert abs(y.iloc[0] - 0.05) < 1e-12            # the 2014 dividend is visible on day one
    assert abs(y.loc["2015-06-30"] - 0.11) < 1e-12  # both inside the trailing year: 1.1/10
    assert abs(y.loc["2016-02-01"] - 0.06) < 1e-12  # the 2014 event has aged out
