"""Batch ingestion: BaoStock -> adjusted parquet data lake.

First-milestone scope: the **HS300** universe (free, via BaoStock), 前复权 daily
bars over the backtest window, persisted one parquet per symbol plus a universe
manifest tagged with board + price-limit rule.

KNOWN LIMITATION (logged, never hidden): BaoStock's hs300 query returns *current*
constituents, so this first cut carries **survivorship bias**. Point-in-time
membership (snapshot per rebalance, including removed names) and the full delisted
universe are the next data-quality milestone — see TODO at the bottom.
"""
from __future__ import annotations

import baostock as bs
import pandas as pd

from ..paths import PARQUET_DIR, RAW_DIR, ensure_dirs
from .sources import baostock_source as bss

BACKTEST_START = "2015-01-01"
BACKTEST_END = "2025-12-31"

# Per-board daily price-limit (涨跌停) magnitude — needed by the friction-faithful
# backtest gate (orders at the limit must not fill).
PRICE_LIMIT_PCT = {"Main": 0.10, "STAR": 0.20, "ChiNext": 0.20, "BSE": 0.30}


def board_of(code: str) -> str:
    """Map a BaoStock code ('sh.600000') to its board, for price-limit rules."""
    num = code.split(".")[-1]
    if num.startswith("688"):
        return "STAR"          # 科创板 ±20%
    if num.startswith(("300", "301")):
        return "ChiNext"       # 创业板 ±20%
    if num.startswith(("4", "8", "920")):
        return "BSE"           # 北交所 ±30%
    return "Main"              # 主板 ±10%


def _rs_to_df(rs) -> pd.DataFrame:
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def hs300_universe(day: str | None = None) -> pd.DataFrame:
    """Current HS300 constituents via BaoStock. Call inside a `bss.session()`.

    Returns columns: code, code_name, board, price_limit_pct.
    """
    rs = bs.query_hs300_stocks(date=day)
    if rs.error_code != "0":
        raise RuntimeError(f"hs300 query failed: {rs.error_code} {rs.error_msg}")
    df = _rs_to_df(rs)
    df["board"] = df["code"].map(board_of)
    df["price_limit_pct"] = df["board"].map(PRICE_LIMIT_PCT)
    return df


def pull_universe(codes, start: str = BACKTEST_START, end: str = BACKTEST_END) -> pd.DataFrame:
    """Pull 前复权 daily bars per code -> data/parquet/daily/<code>.parquet.

    Manages its own BaoStock session. Records per-symbol status and continues on
    error. Returns a summary DataFrame (code, rows, status).
    """
    ensure_dirs()
    out = PARQUET_DIR / "daily"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    n = len(codes)
    with bss.session():
        for i, code in enumerate(codes, 1):
            try:
                df = bss.daily_bars(code, start, end, adjustflag="2")
                if not df.empty:
                    df.to_parquet(out / f"{code.replace('.', '_')}.parquet", index=False)
                results.append({"code": code, "rows": len(df), "status": "ok"})
            except Exception as exc:  # noqa: BLE001 — record and keep the batch going
                results.append({"code": code, "rows": 0, "status": f"error: {exc}"})
            if i % 25 == 0 or i == n:
                print(f"  ...{i}/{n} pulled")
    return pd.DataFrame(results)


def ingest_hs300(start: str = BACKTEST_START, end: str = BACKTEST_END,
                 limit: int | None = None) -> pd.DataFrame:
    """End-to-end: resolve HS300 -> pull bars -> write manifest + summary.

    `limit` caps the symbol count for a quick smoke test (logged when used).
    """
    ensure_dirs()
    with bss.session():
        uni = hs300_universe()
    if limit is not None:
        print(f"[SMOKE] limiting to first {limit} of {len(uni)} HS300 symbols")
        uni = uni.head(limit)
    print(f"HS300 universe: {len(uni)} symbols "
          f"(survivorship caveat: CURRENT membership only -- see module docstring)")
    uni.to_csv(RAW_DIR / "hs300_universe.csv", index=False)

    summary = pull_universe(uni["code"].tolist(), start, end)
    ok = int((summary["status"] == "ok").sum())
    print(f"pulled {ok}/{len(summary)} symbols -> {PARQUET_DIR / 'daily'}")
    summary.to_csv(RAW_DIR / "hs300_pull_summary.csv", index=False)
    return summary


# TODO(next data milestone): point-in-time HS300 membership (snapshot per rebalance,
# keep removed names) + full delisted universe via Tushare stock_basic(list/delist
# dates) to kill survivorship bias before any results are trusted.
