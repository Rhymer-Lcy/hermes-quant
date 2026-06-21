"""Point-in-time HS300 index membership (BaoStock, free).

This is the survivorship-bias fix: instead of backtesting today's HS300 members
over the past, we snapshot the constituents AS OF each month-end (including names
later removed or delisted), and the backtest selects only from the then-current
set. `query_hs300_stocks(date)` returns the constituents effective on that date.
"""
from __future__ import annotations

import bisect
from typing import Any

import baostock as bs
import pandas as pd

from ..io import atomic_to_parquet
from ..paths import PARQUET_DIR, RAW_DIR, ensure_dirs
from .ingest import BACKTEST_END, BACKTEST_START
from .sources import baostock_source as bss

MEMBERSHIP_PARQUET = PARQUET_DIR / "hs300_membership.parquet"
UNION_CSV = RAW_DIR / "hs300_union.csv"
CSI500_MEMBERSHIP_PARQUET = PARQUET_DIR / "csi500_membership.parquet"
CSI500_UNION_CSV = RAW_DIR / "csi500_union.csv"


def rs_to_df(rs: Any) -> pd.DataFrame:
    """Drain a BaoStock result set into a DataFrame. Public so the live feed can reuse it."""
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def month_end_trading_dates(start: str, end: str) -> list[str]:
    """Last actual trading day of each month in [start, end]. Call inside session()."""
    rs = bs.query_trade_dates(start_date=start, end_date=end)
    cal = rs_to_df(rs)
    cal = cal[cal["is_trading_day"] == "1"].copy()
    cal["d"] = pd.to_datetime(cal["calendar_date"])
    last = cal.groupby(cal["d"].dt.to_period("M"))["calendar_date"].last()
    return last.tolist()


def _build_index_membership(query_fn, parquet_path, union_csv, label: str,
                            start: str, end: str) -> tuple[pd.DataFrame, list[str]]:
    """Snapshot month-end index membership via `query_fn(date=)` (e.g. bs.query_hs300_stocks
    or bs.query_zz500_stocks) -> persist a (date, code) table + the survivorship-free union of
    all names ever in the index over the window. Returns (table, union)."""
    ensure_dirs()
    rows = []
    with bss.session():
        for d in month_end_trading_dates(start, end):
            df = rs_to_df(query_fn(date=d))
            rows.extend({"date": d, "code": c} for c in df["code"].tolist())
    mdf = pd.DataFrame(rows)
    mdf["date"] = pd.to_datetime(mdf["date"])
    atomic_to_parquet(mdf, parquet_path, index=False)
    union = sorted(mdf["code"].unique())
    pd.Series(union, name="code").to_csv(union_csv, index=False)
    print(f"{label}: {mdf['date'].nunique()} monthly snapshots, {len(union)} unique names ever in index")
    return mdf, union


def build_membership(start: str = BACKTEST_START, end: str = BACKTEST_END) -> tuple[pd.DataFrame, list[str]]:
    """Snapshot month-end HS300 (CSI 300) membership + its all-time union. Returns (table, union)."""
    return _build_index_membership(bs.query_hs300_stocks, MEMBERSHIP_PARQUET, UNION_CSV, "HS300", start, end)


def build_csi500_membership(start: str = BACKTEST_START, end: str = BACKTEST_END) -> tuple[pd.DataFrame, list[str]]:
    """Snapshot month-end CSI500 (CSI 500) membership + its all-time union (survivorship-free,
    free + PIT via BaoStock query_zz500_stocks). Returns (table, union)."""
    return _build_index_membership(bs.query_zz500_stocks, CSI500_MEMBERSHIP_PARQUET, CSI500_UNION_CSV, "CSI500", start, end)


def membership_lookup(mdf: pd.DataFrame):
    """Return f(date)->set(codes): HS300 members as of the latest snapshot <= date."""
    snaps = {d: set(g["code"]) for d, g in mdf.groupby("date")}
    snap_dates = sorted(snaps)

    def asof(when) -> set[str]:
        when = pd.Timestamp(when)
        i = bisect.bisect_right(snap_dates, when) - 1
        return snaps[snap_dates[i]] if i >= 0 else set()

    return asof
