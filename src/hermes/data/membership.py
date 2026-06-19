"""Point-in-time HS300 index membership (BaoStock, free).

This is the survivorship-bias fix: instead of backtesting today's HS300 members
over the past, we snapshot the constituents AS OF each month-end (including names
later removed or delisted), and the backtest selects only from the then-current
set. `query_hs300_stocks(date)` returns the constituents effective on that date.
"""
from __future__ import annotations

import bisect

import baostock as bs
import pandas as pd

from ..paths import PARQUET_DIR, RAW_DIR, ensure_dirs
from .ingest import BACKTEST_END, BACKTEST_START
from .sources import baostock_source as bss

MEMBERSHIP_PARQUET = PARQUET_DIR / "hs300_membership.parquet"
UNION_CSV = RAW_DIR / "hs300_union.csv"


def _rs_to_df(rs) -> pd.DataFrame:
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def month_end_trading_dates(start: str, end: str) -> list[str]:
    """Last actual trading day of each month in [start, end]. Call inside session()."""
    rs = bs.query_trade_dates(start_date=start, end_date=end)
    cal = _rs_to_df(rs)
    cal = cal[cal["is_trading_day"] == "1"].copy()
    cal["d"] = pd.to_datetime(cal["calendar_date"])
    last = cal.groupby(cal["d"].dt.to_period("M"))["calendar_date"].last()
    return last.tolist()


def build_membership(start: str = BACKTEST_START, end: str = BACKTEST_END) -> tuple[pd.DataFrame, list[str]]:
    """Snapshot month-end HS300 membership -> persist a (date, code) table and the
    union of all names ever in HS300 over the window. Returns (table, union)."""
    ensure_dirs()
    rows = []
    with bss.session():
        dates = month_end_trading_dates(start, end)
        for d in dates:
            rs = bs.query_hs300_stocks(date=d)
            df = _rs_to_df(rs)
            for code in df["code"].tolist():
                rows.append({"date": d, "code": code})

    mdf = pd.DataFrame(rows)
    mdf["date"] = pd.to_datetime(mdf["date"])
    mdf.to_parquet(MEMBERSHIP_PARQUET, index=False)
    union = sorted(mdf["code"].unique())
    pd.Series(union, name="code").to_csv(UNION_CSV, index=False)
    print(f"membership: {mdf['date'].nunique()} monthly snapshots, "
          f"{len(union)} unique names ever in HS300 (vs 300 current)")
    return mdf, union


def membership_lookup(mdf: pd.DataFrame):
    """Return f(date)->set(codes): HS300 members as of the latest snapshot <= date."""
    snaps = {d: set(g["code"]) for d, g in mdf.groupby("date")}
    snap_dates = sorted(snaps)

    def asof(when) -> set[str]:
        when = pd.Timestamp(when)
        i = bisect.bisect_right(snap_dates, when) - 1
        return snaps[snap_dates[i]] if i >= 0 else set()

    return asof
