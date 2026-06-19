"""BaoStock -> adjusted parquet data lake: daily-bar ingestion utilities.

`pull_universe` pulls 前复权 daily bars for a list of codes into data/parquet/daily/.
The survivorship-bias fix (point-in-time HS300 membership and the union of all names
ever in the index) lives in hermes.data.membership; the union it produces is the
universe pulled here. There is intentionally no current-membership pull -- that would
reintroduce survivorship bias.
"""
from __future__ import annotations

import pandas as pd

from ..paths import PARQUET_DIR, RAW_DIR, ensure_dirs
from .sources import baostock_source as bss

BACKTEST_START = "2015-01-01"
BACKTEST_END = "2025-12-31"

# Per-board daily price-limit (涨跌停) magnitude -- needed by the friction-faithful
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
            except Exception as exc:  # noqa: BLE001 -- record and keep the batch going
                results.append({"code": code, "rows": 0, "status": f"error: {exc}"})
            if i % 25 == 0 or i == n:
                print(f"  ...{i}/{n} pulled")
    return pd.DataFrame(results)


def write_pull_summary(summary: pd.DataFrame, name: str = "pull") -> None:
    """Persist a pull summary CSV to data/raw/ and print the error rows, if any."""
    ensure_dirs()
    summary.to_csv(RAW_DIR / f"{name}_pull_summary.csv", index=False)
    ok = int((summary["status"] == "ok").sum())
    print(f"pulled {ok}/{len(summary)} symbols -> {PARQUET_DIR / 'daily'}")
    errors = summary[summary["status"] != "ok"]
    if not errors.empty:
        print("symbols with errors (likely never-traded / unlisted codes):")
        print(errors.to_string(index=False))
