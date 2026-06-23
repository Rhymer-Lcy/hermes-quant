"""BaoStock adapter — free, anonymous (no token, no registration), API-based.

BaoStock requires a login()/logout() pair around queries; login is anonymous.
Use the `session()` context manager to guarantee logout even on error.

    from hermes.data.sources import baostock_source as bss
    with bss.session():
        df = bss.daily_bars("sh.600000", "2015-01-01", "2025-12-31")

adjustflag: "1" = backward-adjusted, "2" = forward-adjusted, "3" = unadjusted.
"""
from __future__ import annotations

import os
import socket
from contextlib import contextmanager

import baostock as bs
import pandas as pd

# BaoStock's client connects by HOSTNAME (despite the name, cons.BAOSTOCK_SERVER_IP is
# "public-api.baostock.com"); see _resolve_server_ip for why we pin it to an IP. Reaching into the
# vendor's internal constants is guarded: if a future layout change hides them, fall back to the
# known host/port and skip pinning (login then uses BaoStock's own DNS path, as before).
try:
    import baostock.common.contants as _bs_cons

    _SERVER_HOST = _bs_cons.BAOSTOCK_SERVER_IP            # "public-api.baostock.com"
    _SERVER_PORT = _bs_cons.BAOSTOCK_SERVER_PORT          # 10030
except Exception:                                        # pragma: no cover - vendor internals moved
    _bs_cons = None
    _SERVER_HOST, _SERVER_PORT = "public-api.baostock.com", 10030

_FALLBACK_SERVER_IP = "114.94.20.73"                     # public-api.baostock.com (seed; cache refreshes it)

# Full daily field set BaoStock exposes for stocks.
_DAILY_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST,adjustflag"
)

_NUMERIC = [
    "open", "high", "low", "close", "preclose", "volume", "amount",
    "turn", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM",
]


class BaoStockUnavailable(RuntimeError):
    """BaoStock cannot currently supply a complete, current dataset -- either it is unreachable
    (network/transport failure) or it has not finished posting the latest day's data. This is
    TRANSIENT: a later retry, once connectivity is restored or publication completes, succeeds.
    The unattended driver maps it to a distinct exit code so the wrapper retries with backoff
    instead of failing the day (unlike a malformed-query error or an integrity-gate failure)."""


# BaoStock reports transport problems via the 10002001-10002008 "network error/connect/send/recv"
# code family (all phrased with '网络' = network) -- transient, unlike a malformed-query or auth
# error. Classify them so the driver can retry rather than abort. The message substring is a
# backstop in case the vendor adds an unlisted network code.
_NETWORK_ERROR_CODES = frozenset(f"1000200{n}" for n in range(1, 9))


def _is_network_error(error_code: str, error_msg: str) -> bool:
    """True if a BaoStock error is a transient transport failure (retryable). Matches the known
    10002001-10002008 network-error family or a '网络' (network) phrasing in the message, so an
    unlisted network code is still caught by the message. Pure -- unit-tested without a session."""
    return error_code in _NETWORK_ERROR_CODES or "网络" in (error_msg or "")


def _server_ip_cache_path():
    from ...paths import CACHE_DIR
    return CACHE_DIR / "baostock_server_ip.txt"


def _read_cached_server_ip() -> str | None:
    try:
        p = _server_ip_cache_path()
        return (p.read_text(encoding="utf-8").strip() or None) if p.exists() else None
    except OSError:
        return None


def _cache_server_ip(ip: str) -> None:
    try:
        from ...io import atomic_write_text
        from ...paths import CACHE_DIR
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_text(ip, _server_ip_cache_path())
    except OSError:
        pass                                             # a stale/absent cache just falls back below


def _resolve_server_ip(host: str = _SERVER_HOST) -> str:
    """Resolve BaoStock's server hostname to an IPv4 address, robust to a VPN that breaks the
    DEFAULT resolver while leaving the server's physical route intact (observed with Sangfor SSL
    VPN: getaddrinfo on public-api.baostock.com fails, yet the IP is directly reachable). Order:
    HERMES_BAOSTOCK_IP override (the wrapper resolves via a public DNS server and sets it) -> OS
    resolution, cached on success -> last cached IP -> seed. The result is handed to the socket
    directly, so login no longer needs DNS."""
    override = os.environ.get("HERMES_BAOSTOCK_IP", "").strip()
    if override:
        _cache_server_ip(override)                       # keep the cache fresh for manual runs too
        return override
    try:
        ip = socket.getaddrinfo(host, _SERVER_PORT, family=socket.AF_INET,
                                type=socket.SOCK_STREAM)[0][4][0]
        _cache_server_ip(ip)
        return ip
    except OSError:                                      # VPN-broken DNS, offline, etc.
        return _read_cached_server_ip() or _FALLBACK_SERVER_IP


def _pin_server_ip() -> str | None:
    """Point the BaoStock client's socket at a resolved IP (idempotent). No-op if the vendor's
    constants module could not be imported. Returns the IP pinned (or None)."""
    if _bs_cons is None:
        return None
    ip = _resolve_server_ip()
    _bs_cons.BAOSTOCK_SERVER_IP = ip
    return ip


@contextmanager
def session():
    """Anonymous BaoStock session. No account/credentials needed. The server hostname is resolved
    and pinned to an IP first (see _pin_server_ip), so login works even behind a VPN that breaks
    DNS. A login failure from a transport problem raises BaoStockUnavailable (retryable); any other
    login failure raises a plain RuntimeError (fatal)."""
    _pin_server_ip()
    lg = bs.login()
    if lg.error_code != "0":
        msg = f"BaoStock login failed: {lg.error_code} {lg.error_msg}"
        if _is_network_error(lg.error_code, lg.error_msg):
            raise BaoStockUnavailable(msg)
        raise RuntimeError(msg)
    try:
        yield
    finally:
        bs.logout()


def daily_bars(code: str, start: str, end: str, adjustflag: str = "2") -> pd.DataFrame:
    """Daily bars for one symbol. Call inside `session()`. Returns a typed DataFrame.

    `code` is BaoStock-style: 'sh.600000', 'sz.000001'.
    """
    rs = bs.query_history_k_data_plus(
        code, _DAILY_FIELDS, start_date=start, end_date=end,
        frequency="d", adjustflag=adjustflag,
    )
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock query failed for {code}: {rs.error_code} {rs.error_msg}")

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())

    df = pd.DataFrame(rows, columns=rs.fields)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for col in _NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "isST" in df.columns:
        df["isST"] = df["isST"].astype("string").fillna("0").eq("1")
    return df


def stock_industry(date: str | None = None) -> pd.DataFrame:
    """Shenwan industry classification, FREE. Call inside `session()`.

    `date=None` returns the latest snapshot; `date='2020-06-30'` returns the snapshot
    effective at/just before that date (DATE-AWARE -> point-in-time capable). Columns:
    updateDate, code, code_name, industry, industryClassification. `code` is BaoStock-style
    ('sh.600000'), so it joins the rest of the lake directly; `industry` can be empty for a
    few delisted/untagged names. Note: Chinese `industry`/`code_name` strings are returned
    GBK-encoded -- group/key on the raw string, never on a re-decoded console rendering.

    Useful for sector EXPOSURE/attribution (not as an alpha lever: sector-neutralizing value
    in HS300 worsens drawdown -- see docs/risk_control.md A3)."""
    rs = bs.query_stock_industry(date=date) if date else bs.query_stock_industry()
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock industry query failed: {rs.error_code} {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def index_close(code: str, start: str, end: str) -> pd.Series:
    """Daily close for an index (e.g. 'sh.000300' = CSI 300). Call inside `session()`.
    Indices have no valuation/adjustment fields, so only date+close are requested."""
    rs = bs.query_history_k_data_plus(code, "date,close", start_date=start, end_date=end,
                                      frequency="d")
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock index query failed for {code}: {rs.error_code} {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    if df.empty:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    return pd.to_numeric(df["close"], errors="coerce").set_axis(df["date"]).rename(code)
