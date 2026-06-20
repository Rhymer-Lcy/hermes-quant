"""Intraday data adapters (AKShare, free, no token). Isolated to the intraday line.

AKShare is a scraper, so: no account/token, but treat outputs as best-effort -- shallow history
and occasional schema drift. Futures minute bars come from Sina (reachable, validated); stock and
convertible-bond minute bars come from EastMoney (may be proxy-blocked in some environments).
"""
from __future__ import annotations

import pandas as pd

_FUTURES_NUM = ["open", "high", "low", "close", "volume", "hold"]


def futures_minute(symbol: str, period: str = "5") -> pd.DataFrame:
    """Sina intraday bars for one futures contract, e.g. 'IF2609' (沪深300 期货), 'IC2609'
    (中证500 期货), 'rb2610' (螺纹钢). `period` in {'1','5','15','30','60'} minutes.

    Returns a datetime-indexed frame: [open, high, low, close, volume, hold(持仓量)]. FREE, no
    token. Validated: futures_zh_minute_sina('IF2609','5') -> 1023 rows. CAVEAT: Sina serves only
    a SHALLOW recent window (~1000+ bars), not deep history -- enough to prototype signals; deep
    minute/tick history needs a paid vendor."""
    import akshare as ak
    df = ak.futures_zh_minute_sina(symbol=symbol, period=period)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df[_FUTURES_NUM] = df[_FUTURES_NUM].apply(pd.to_numeric, errors="coerce")
    return df.set_index("datetime").sort_index()


def cb_minute(symbol: str, period: str = "5") -> pd.DataFrame:
    """Convertible-bond (可转债, T+0) intraday bars, e.g. 'sh113537'. `period` in
    {'1','5','15','30','60'}. FREE, no token (EastMoney via AKShare -- may be proxy-blocked in
    sandboxed environments; works on a normal connection). Returns a datetime-indexed OHLCV frame."""
    import akshare as ak
    df = ak.bond_zh_hs_cov_min(symbol=symbol, period=period)
    cols = {c: c for c in df.columns}
    df = df.rename(columns={"时间": "datetime"}) if "时间" in df.columns else df.rename(columns={df.columns[0]: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.set_index("datetime").sort_index()
