"""Intraday data adapters (AKShare, free, no token). Isolated to the intraday line.

AKShare is a scraper, so: no account/token, but treat outputs as best-effort -- shallow history
and occasional schema drift. Futures minute bars come from Sina (reachable, validated); stock and
convertible-bond minute bars come from EastMoney (may be proxy-blocked in some environments).
"""
from __future__ import annotations

import pandas as pd

from ..io import atomic_to_parquet
from ..paths import PARQUET_DIR

_FUTURES_NUM = ["open", "high", "low", "close", "volume", "hold"]
INTRADAY_DIR = PARQUET_DIR / "intraday"


def futures_minute(symbol: str, period: str = "5") -> pd.DataFrame:
    """Sina intraday bars for one futures contract, e.g. 'IF2609' (CSI 300 futures), 'IC2609'
    (CSI 500 futures), 'rb2610' (rebar). `period` in {'1','5','15','30','60'} minutes.

    Returns a datetime-indexed frame: [open, high, low, close, volume, hold (open interest)]. FREE, no
    token. Validated: futures_zh_minute_sina('IF2609','5') -> 1023 rows. CAVEAT: Sina serves only
    a SHALLOW recent window (~1000+ bars), not deep history -- enough to prototype signals; deep
    minute/tick history needs a paid vendor."""
    import akshare as ak
    df = ak.futures_zh_minute_sina(symbol=symbol, period=period)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df[_FUTURES_NUM] = df[_FUTURES_NUM].apply(pd.to_numeric, errors="coerce")
    return df.set_index("datetime").sort_index()


def accumulate_futures_minute(symbol: str = "IF0", periods: tuple[str, ...] = ("1", "5")) -> dict[str, int]:
    """Pull the recent Sina minute window for `symbol` at each `period` and UNION it into a growing
    parquet at data/parquet/intraday/<symbol>_<period>m.parquet. Sina only serves a shallow recent
    window (~1023 bars), so a DAILY run accumulates history forward; overlapping windows dedupe on
    `datetime` (idempotent -- safe to re-run / miss a day within the lookback). Returns {period: total_rows}.

    Default 1m (finest; ~5-trading-day lookback so a daily job never gaps) + 5m (~22-day lookback, a
    gap-robust backbone). Cheap: a few sub-second calls + small atomic writes -- negligible resources."""
    INTRADAY_DIR.mkdir(parents=True, exist_ok=True)
    totals: dict[str, int] = {}
    for p in periods:
        new = futures_minute(symbol, p).reset_index()          # datetime + OHLC/volume/hold
        path = INTRADAY_DIR / f"{symbol}_{p}m.parquet"
        if path.exists():
            new = (pd.concat([pd.read_parquet(path), new])
                   .drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True))
        atomic_to_parquet(new, path, index=False)
        totals[p] = len(new)
    return totals


def cb_minute(symbol: str, period: str = "5") -> pd.DataFrame:
    """Convertible-bond (T+0) intraday bars, e.g. 'sh113537'. `period` in
    {'1','5','15','30','60'}. FREE, no token (EastMoney via AKShare -- may be proxy-blocked in
    sandboxed environments; works on a normal connection). Returns a datetime-indexed OHLCV frame."""
    import akshare as ak
    df = ak.bond_zh_hs_cov_min(symbol=symbol, period=period)
    df = df.rename(columns={"时间": "datetime"}) if "时间" in df.columns else df.rename(columns={df.columns[0]: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.set_index("datetime").sort_index()
