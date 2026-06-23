"""Live EOD data refresh for paper trading -- thin wrappers over the SAME ingestion the
research lake uses (BaoStock), so paper data == research data (no source skew).

Two refreshes, run after market close on a trading day:
  - extend_membership(): pull HS300 month-end snapshots NEWER than the last stored one and
    append; rebuild the all-time union (so 2026 entrants get added without losing history).
  - update_daily_bars(): incrementally refresh forward-adjusted daily bars for the union through `end`.

INCREMENTAL refresh, re-base-safe: forward-adjusted prices are RE-BASED across the entire history
whenever a dividend/split occurs, so a blind append would mix two bases in one series. Instead each
code appends only its new bars UNLESS its overlap bar's close changed on re-pull (which signals a
re-base) -- those few names are re-pulled in full (see ingest.pull_universe_incremental). This keeps
the whole lake on ONE consistent basis while turning the daily run from a ~40-min full re-pull into
minutes (and a no-op re-run into seconds); live.paper then recomputes the ledger wholesale from the
seed, so the equity curve stays self-consistent.
"""
from __future__ import annotations

from datetime import date

import baostock as bs
import pandas as pd

from ..data import ingest
from ..data.lake import load_close_panel
from ..data.membership import (MEMBERSHIP_PARQUET, UNION_CSV,
                               month_end_trading_dates, rs_to_df)
from ..data.sources import baostock_source as bss
from ..data.sources.baostock_source import BaoStockUnavailable
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
    # Avoid concatenating an empty `new` (the common "already current" case): pandas warns on a
    # concat involving empty/all-NA frames (its dtype handling is changing), and the result is
    # simply `existing` anyway.
    if existing is None:
        mdf = new
    elif new.empty:
        mdf = existing
    else:
        mdf = pd.concat([existing, new], ignore_index=True)
    mdf = mdf.drop_duplicates(["date", "code"]).sort_values(["date", "code"]).reset_index(drop=True)
    atomic_to_parquet(mdf, MEMBERSHIP_PARQUET, index=False)
    union = sorted(mdf["code"].unique())
    ensure_dirs()
    pd.Series(union, name="code").to_csv(UNION_CSV, index=False)
    print(f"membership: +{len(new_dates)} new snapshot(s), {len(union)} names in union "
          f"(through {end})")
    return mdf, union, new_dates


def assert_pull_healthy(summary: pd.DataFrame, n_union: int, min_ok_fraction: float = 0.98) -> float:
    """Return the OK fraction of a pull summary; raise BaoStockUnavailable (transient, retryable)
    if it falls below `min_ok_fraction` (a degraded pull would leave a mixed-basis lake -- see
    update_daily_bars)."""
    ok = int((summary["status"] == "ok").sum()) if len(summary) else 0
    frac = ok / n_union if n_union else 0.0
    if frac < min_ok_fraction:
        raise BaoStockUnavailable(                     # transient (connectivity): retryable with backoff
            f"degraded BaoStock pull: {ok}/{n_union} ok ({frac:.1%} < {min_ok_fraction:.0%}); "
            "refusing to update the live record on a partial/mixed-basis lake (re-run when the "
            "source recovers -- the next full pull self-heals)")
    return frac


def update_daily_bars(union: list[str], end: str | None = None,
                      min_ok_fraction: float = 0.98) -> pd.DataFrame:
    """INCREMENTALLY refresh forward-adjusted daily bars for `union` through `end` (appends new bars;
    full-re-pulls only re-based names -- see ingest.pull_universe_incremental). Returns the summary.

    DATA-INTEGRITY GATE (for unattended daily operation): a BaoStock pull can have names time out
    mid-batch, leaving them stale while the rest advance -- a mixed lake. The pull records-and-
    continues per code, so here we RAISE if the OK fraction drops below `min_ok_fraction`. Raising
    aborts refresh() before any live report is written, so the auto-maintained record never computes
    on a degraded lake; the next clean run self-heals."""
    end = end or _today()
    summary = ingest.pull_universe_incremental(union, ingest.BACKTEST_START, end)
    ingest.write_pull_summary(summary, name="live_refresh")
    assert_pull_healthy(summary, len(union), min_ok_fraction)
    return summary


def latest_coverage(panel: pd.DataFrame, members: list[str]) -> tuple[pd.Timestamp, float]:
    """(lake's latest date, fraction of `members` that carry a bar on it). Pure and testable."""
    latest = panel.index[-1]
    cols = [c for c in members if c in panel.columns]
    cov = float(panel.loc[latest, cols].notna().mean()) if cols else 0.0
    return latest, cov


def current_members(mdf: pd.DataFrame) -> list[str]:
    """The most recent membership snapshot's codes (the names that should all trade today)."""
    return sorted(mdf[mdf["date"] == mdf["date"].max()]["code"].unique())


def assert_publication_complete(members: list[str], min_coverage: float = 0.90) -> float:
    """Guard against INCOMPLETE same-day publication. BaoStock posts EOD bars over ~2-3 hours
    after the close, so a run inside that window finds today's bar for only some names while the
    rest still end on the prior day; the backtest's right-edge rule would then mistake the
    not-yet-posted names for delistings and force-liquidate them. If fewer than `min_coverage` of
    the CURRENT index members carry a bar on the lake's latest date, raise BaoStockUnavailable
    (transient, retryable) -- refusing to compute on a half-published day (the next run, once
    publication completes, self-heals). The fixed-time schedule already targets the
    post-publication window; this is the fail-loud backstop."""
    latest, cov = latest_coverage(load_close_panel(codes=members, field="close"), members)
    if cov < min_coverage:
        raise BaoStockUnavailable(                     # transient (mid-publication): retryable with backoff
            f"incomplete publication: only {cov:.0%} of current members carry a {latest.date()} "
            f"bar (< {min_coverage:.0%}); BaoStock is still posting today's EOD data. Refusing to "
            "update (would mis-liquidate the unposted names); retry when publication completes.")
    return cov


def refresh(end: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """One call: extend membership to `end`, refresh the union's daily bars, and verify the day's
    publication is complete -- each failing loud (see update_daily_bars / assert_publication_complete)
    before any live report is written. Returns (membership_df, union). Run after BaoStock has posted
    the day's EOD data (~2-3 h after close; the task is scheduled in the evening)."""
    end = end or _today()
    mdf, union, _ = extend_membership(end)
    update_daily_bars(union, end)
    assert_publication_complete(current_members(mdf))
    return mdf, union
