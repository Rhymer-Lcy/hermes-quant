"""BaoStock adapter — free, anonymous (no token, no registration), API-based.

BaoStock requires a login()/logout() pair around queries; login is anonymous.
Use the `session()` context manager to guarantee logout even on error.

    from hermes.data.sources import baostock_source as bss
    with bss.session():
        df = bss.daily_bars("sh.600000", "2015-01-01", "2025-12-31")

adjustflag: "1" = backward-adjusted, "2" = forward-adjusted, "3" = unadjusted.
"""
from __future__ import annotations

from contextlib import contextmanager

import baostock as bs
import pandas as pd

# Full daily field set BaoStock exposes for stocks.
_DAILY_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST,adjustflag"
)

_NUMERIC = [
    "open", "high", "low", "close", "preclose", "volume", "amount",
    "turn", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM",
]


@contextmanager
def session():
    """Anonymous BaoStock session. No account/credentials needed."""
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {lg.error_code} {lg.error_msg}")
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
