"""Pure data helpers: board mapping, ts_code conversion, PIT membership as-of."""
import pandas as pd

from hermes.data.ingest import PRICE_LIMIT_PCT, _is_rebased, board_of
from hermes.data.membership import membership_lookup
from hermes.data.sources.baostock_source import _is_network_error
from hermes.data.sources.tushare_source import to_ts_code


def test_board_of_mapping():
    assert board_of("sh.688981") == "STAR"      # STAR Market
    assert board_of("sz.300750") == "ChiNext"   # ChiNext
    assert board_of("sz.301234") == "ChiNext"
    assert board_of("bj.830799") == "BSE"        # 8-prefix Beijing Stock Exchange
    assert board_of("bj.430047") == "BSE"        # 4-prefix
    assert board_of("sh.600000") == "Main"       # Main Board
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


def test_is_rebased_detects_dividend_only():
    # incremental refresh appends unless the forward-adjusted series re-based (dividend/split)
    assert _is_rebased(10.0, 9.70) is True            # ex-dividend rescaled the whole history
    assert _is_rebased(10.0, 10.0) is False           # unchanged -> safe to append the new bars
    assert _is_rebased(10.0, 10.0 + 1e-9) is False    # float noise, not a re-base


def test_is_network_error_classifies_transient_failures():
    # transient transport failures -> retryable (driver exits EX_TEMPFAIL; wrapper retries):
    assert _is_network_error("10002001", "网络错误") is True       # family floor (BSERR_SOCKET_ERR)
    assert _is_network_error("10002007", "网络接收错误") is True   # the observed VPN-time failure
    assert _is_network_error("10002008", "") is True              # family ceiling, empty message
    assert _is_network_error("10009999", "网络连接错误") is True   # unlisted code, caught by message
    # non-network failures -> fatal (no retry):
    assert _is_network_error("0", "success") is False             # not an error at all
    assert _is_network_error("10001002", "用户名或密码错误") is False  # auth error, not transport
    assert _is_network_error("10004001", None) is False           # parse error, missing message


def test_resolve_server_ip_fallback_chain(monkeypatch):
    # The server hostname is resolved + pinned so login survives a VPN that breaks DNS. Order:
    # env override -> OS resolution -> last cached IP -> seed.
    import socket as _socket

    from hermes.data.sources import baostock_source as bss

    monkeypatch.setattr(bss, "_cache_server_ip", lambda ip: None)   # never touch disk in the test

    # 1) explicit override (the wrapper resolves via public DNS and sets it) wins
    monkeypatch.setenv("HERMES_BAOSTOCK_IP", "1.2.3.4")
    assert bss._resolve_server_ip() == "1.2.3.4"

    # 2) no override + DNS broken (VPN) -> last cached IP
    monkeypatch.delenv("HERMES_BAOSTOCK_IP", raising=False)
    monkeypatch.setattr(_socket, "getaddrinfo", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(bss, "_read_cached_server_ip", lambda: "5.6.7.8")
    assert bss._resolve_server_ip() == "5.6.7.8"

    # 3) no override + DNS broken + no cache -> seed
    monkeypatch.setattr(bss, "_read_cached_server_ip", lambda: None)
    assert bss._resolve_server_ip() == bss._FALLBACK_SERVER_IP
