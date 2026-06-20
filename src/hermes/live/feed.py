"""Live EOD data refresh for paper trading -- thin wrappers over the SAME ingestion the
research lake uses (BaoStock), so paper data == research data (no source skew).

Two refreshes, run after market close on a trading day:
  - extend_membership(): pull HS300 month-end snapshots NEWER than the last stored one and
    append; rebuild the all-time union (so 2026 entrants get added without losing history).
  - update_daily_bars(): re-pull 前复权 daily bars for the union through `end`.

WHY a FULL re-pull, not an append: 前复权 (forward-adjusted) prices are RE-BASED across the
entire history whenever a dividend/split occurs, so appending only new days would mix two
adjustment bases in one series. A full overwrite (pull_universe already overwrites per code)
keeps the whole lake on ONE consistent basis; live.paper then recomputes the ledger wholesale
from the seed, so the equity curve is always self-consistent. Cost: a few minutes of free
BaoStock calls per run -- fine for a monthly strategy refreshed once a trading day. (A
trailing-window incremental would need 不复权 + an adjustment factor stored separately; deferred.)
"""
from __future__ import annotations

from datetime import date

import baostock as bs
import pandas as pd

from ..data import ingest
from ..data.membership import (MEMBERSHIP_PARQUET, UNION_CSV,
                               month_end_trading_dates, rs_to_df)
from ..data.sources import baostock_source as bss
from ..io import atomic_to_parquet
from ..paths import RAW_DIR, ensure_dirs


def _today() -> str:
    return date.today().strftime("%Y-%m-%d")


def extend_membership(end: str | None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Append HS300 month-end snapshots after the last stored date; rebuild the union.
    Returns (membership_df, union, newly_added_snapshot_dates). Incremental + append-only,
    so existing 2015- history is never rebuilt or lost."""
    end = end or _today()
    existing = pd.read_parquet(MEMBERSHIP_PARQUET) if MEMBERSHIP_PARQUET.exists() else None
    last = existing["date"].max() if existing is not None and len(existing) else None

    # Only snapshot months that have CLOSED: month_end_trading_dates over an incomplete
    # current month would tag the latest available day as a spurious "month-end" and fire a
    # mid-month rebalance. Excluding the current calendar month means the most recent rebalance
    # is end-of-prior-month, executed early this month -- exactly the live monthly cadence.
    cur_month = pd.Timestamp(end).to_period("M")
    rows, new_dates = [], []
    with bss.session():
        for d in month_end_trading_dates(ingest.BACKTEST_START, end):
            if pd.Timestamp(d).to_period("M") >= cur_month:
                continue                                   # current month not closed yet
            if last is not None and pd.Timestamp(d) <= last:
                continue                                   # already stored
            new_dates.append(d)
            snap = rs_to_df(bs.query_hs300_stocks(date=d))
            rows.extend({"date": pd.Timestamp(d), "code": c} for c in snap["code"].tolist())

    new = pd.DataFrame(rows, columns=["date", "code"])
    mdf = pd.concat([existing, new], ignore_index=True) if existing is not None else new
    mdf = mdf.drop_duplicates(["date", "code"]).sort_values(["date", "code"]).reset_index(drop=True)
    atomic_to_parquet(mdf, MEMBERSHIP_PARQUET, index=False)
    union = sorted(mdf["code"].unique())
    ensure_dirs()
    pd.Series(union, name="code").to_csv(UNION_CSV, index=False)
    print(f"membership: +{len(new_dates)} new snapshot(s), {len(union)} names in union "
          f"(through {end})")
    return mdf, union, new_dates


def assert_pull_healthy(summary: pd.DataFrame, n_union: int, min_ok_fraction: float = 0.98) -> float:
    """Return the OK fraction of a pull summary; RAISE if it falls below `min_ok_fraction`
    (a degraded pull would leave a mixed-基准 lake -- see update_daily_bars)."""
    ok = int((summary["status"] == "ok").sum()) if len(summary) else 0
    frac = ok / n_union if n_union else 0.0
    if frac < min_ok_fraction:
        raise RuntimeError(
            f"degraded BaoStock pull: {ok}/{n_union} ok ({frac:.1%} < {min_ok_fraction:.0%}); "
            "refusing to update the live record on a partial/mixed-basis lake (re-run when the "
            "source recovers -- the next full pull self-heals)")
    return frac


def update_daily_bars(union: list[str], end: str | None = None,
                      min_ok_fraction: float = 0.98) -> pd.DataFrame:
    """Full re-pull of 前复权 daily bars for `union` over [BACKTEST_START, end] (overwrites;
    re-basing-safe -- see module docstring). Returns the pull summary.

    DATA-INTEGRITY GATE (for unattended daily operation): a common BaoStock failure is login
    succeeding then names timing out mid-batch, which would leave those names on their prior
    re-basis while the rest are re-based -- a mixed-adjustment lake. pull_universe records-and-
    continues (correct for a one-shot historical ingest), so here we RAISE if the OK fraction
    drops below `min_ok_fraction`. Raising aborts refresh() before any live report is written,
    so the auto-maintained record never computes on a degraded lake; the next clean run re-pulls
    the whole union and self-heals."""
    end = end or _today()
    summary = ingest.pull_universe(union, ingest.BACKTEST_START, end)
    ingest.write_pull_summary(summary, name="live_refresh")
    assert_pull_healthy(summary, len(union), min_ok_fraction)
    return summary


def refresh(end: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """One call: extend membership to `end`, then refresh the union's daily bars (failing loud
    on a degraded pull -- see update_daily_bars). Returns (membership_df, union). Run after
    market close on a trading day."""
    end = end or _today()
    mdf, union, _ = extend_membership(end)
    update_daily_bars(union, end)
    return mdf, union
