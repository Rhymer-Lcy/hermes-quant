"""Pure data helpers: board mapping, ts_code conversion, PIT membership as-of."""
import pandas as pd

from hermes.data.ingest import PRICE_LIMIT_PCT, board_of
from hermes.data.membership import membership_lookup
from hermes.data.sources.tushare_source import to_ts_code


def test_board_of_mapping():
    assert board_of("sh.688981") == "STAR"      # 科创板
    assert board_of("sz.300750") == "ChiNext"   # 创业板
    assert board_of("sz.301234") == "ChiNext"
    assert board_of("bj.830799") == "BSE"        # 8-prefix 北交所
    assert board_of("bj.430047") == "BSE"        # 4-prefix
    assert board_of("sh.600000") == "Main"       # 主板
    assert board_of("sz.000001") == "Main"


def test_price_limit_pct():
    assert PRICE_LIMIT_PCT["Main"] == 0.10
    assert PRICE_LIMIT_PCT["STAR"] == 0.20
    assert PRICE_LIMIT_PCT["ChiNext"] == 0.20
    assert PRICE_LIMIT_PCT["BSE"] == 0.30


def test_to_ts_code():
    assert to_ts_code("sh.600000") == "600000.SH"
    assert to_ts_code("sz.000001") == "000001.SZ"


def test_membership_lookup_asof_semantics():
    mdf = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-31", "2020-01-31", "2020-02-28"]),
        "code": ["a", "b", "c"],
    })
    asof = membership_lookup(mdf)
    assert asof(pd.Timestamp("2019-12-01")) == set()           # before first snapshot
    assert asof(pd.Timestamp("2020-02-01")) == {"a", "b"}      # latest snapshot <= date
    assert asof(pd.Timestamp("2020-02-28")) == {"c"}           # exactly on the new snapshot
    assert asof(pd.Timestamp("2020-06-01")) == {"c"}           # after, carries forward
