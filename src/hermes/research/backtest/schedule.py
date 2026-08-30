"""Rebalance schedules: which bar reads the signal, which bar executes it. Issue #21.

The engine represents a schedule as `{exec_index: signal_index}` over the price panel's bar
positions. `_score_backtest` builds the canonical monthly one internally; this module supplies
alternatives for the calendar-timing study WITHOUT touching that default path -- the engine's
`schedule=` argument is opt-in and `schedule=None` leaves behaviour byte-identical.

The canonical monthly rule is "read the signal at the last trading bar of each calendar period,
execute at the next bar". `calendar_rebalance_schedule(dates, 1)` reproduces it exactly (see the
docstring there), which is what makes the study's D=1 parity gate meaningful rather than
decorative.
"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def month_end_schedule(dates: pd.DatetimeIndex, freq: str = "M") -> dict[int, int]:
    """The canonical rule, extracted verbatim: signal at each period's last bar, execute next bar.

    Kept here so the study can build the baseline through the same code path it tests, rather than
    re-deriving it. `_score_backtest` still computes this inline when `schedule is None`, so the
    deployed path does not depend on this function.
    """
    pos_of = {d: i for i, d in enumerate(dates)}
    n = len(dates)
    period_end = pd.Series(dates, index=dates).groupby(dates.to_period(freq)).max().tolist()
    return {pos_of[sig] + 1: pos_of[sig] for sig in period_end if pos_of[sig] + 1 < n}


def calendar_rebalance_schedule(dates: pd.DatetimeIndex, calendar_day: int) -> dict[int, int]:
    """One rebalance per calendar month, targeted at day `calendar_day`. Frozen in issue #21:

    1. target = min(calendar_day, days_in_month) -- D=31 clamps to 28/29 in February, and D=30/31
       collapse in 30-day months (a real duplicate, reported as such, never deleted).
    2. execution bar = the first trading bar on or after `target` WITHIN THE SAME calendar month;
       if the month has no bar at or after the target (e.g. the 31st falls on a weekend that ends
       the month), the month's LAST trading bar is used instead.
    3. signal bar = the trading bar immediately before the execution bar.
    4. a month whose execution bar is the panel's first bar is skipped -- it has no signal bar,
       and manufacturing one would read the future.

    D=1 is definitionally the canonical rule: the first trading bar on or after the 1st is the
    month's first trading bar, and the bar before it is the previous month's last -- exactly
    "month-end signal, next-bar execution". `month_end_schedule` and this at D=1 therefore agree,
    and the study asserts that rather than assuming it.
    """
    if not 1 <= calendar_day <= 31:
        raise ValueError(f"calendar_day must be in 1..31, got {calendar_day}")
    dates = pd.DatetimeIndex(dates)
    schedule: dict[int, int] = {}
    # positions grouped by calendar month, in bar order
    months: dict[tuple[int, int], list[int]] = {}
    for i, d in enumerate(dates):
        months.setdefault((d.year, d.month), []).append(i)

    for (_y, _m), idx in months.items():
        target = min(calendar_day, dates[idx[0]].days_in_month)
        hit = next((i for i in idx if dates[i].day >= target), idx[-1])
        if hit == 0:                      # no preceding bar -> no signal, skip the month
            continue
        schedule[hit] = hit - 1
    return schedule


def nth_trading_day_schedule(dates: pd.DatetimeIndex, n_th: int) -> dict[int, int]:
    """SECONDARY DIAGNOSTIC (issue #21): execute on the `n_th` trading bar of each month, counting
    from 1, clamped to the month's bar count. Separates a calendar-NUMBER effect from a
    POSITION-in-month effect -- explicitly excluded from the study's verdict."""
    if n_th < 1:
        raise ValueError(f"n_th must be >= 1, got {n_th}")
    dates = pd.DatetimeIndex(dates)
    months: dict[tuple[int, int], list[int]] = {}
    for i, d in enumerate(dates):
        months.setdefault((d.year, d.month), []).append(i)
    schedule: dict[int, int] = {}
    for idx in months.values():
        hit = idx[min(n_th, len(idx)) - 1]
        if hit == 0:
            continue
        schedule[hit] = hit - 1
    return schedule


def schedule_hash(schedule: dict[int, int]) -> str:
    """Stable fingerprint of a schedule, for detecting candidates that resolve identically
    (D=30 and D=31 in a 30-day month, D=29/30/31 in February, ...)."""
    items = tuple(sorted(schedule.items()))
    return f"{hash(items) & 0xFFFFFFFF:08x}"


def assert_no_lookahead(schedule: dict[int, int], dates: pd.DatetimeIndex | None = None) -> None:
    """Every scheduled rebalance must read its signal STRICTLY BEFORE it executes.

    Cheap, and the one invariant whose violation would silently invalidate the whole study, so it
    is asserted rather than trusted. With `dates`, also checks the two bars are adjacent -- the
    engine prices execution at the exec bar's close using a signal read one bar earlier, and a gap
    would mean a stale signal rather than a look-ahead.
    """
    for exec_i, sig_i in schedule.items():
        if sig_i >= exec_i:
            raise AssertionError(f"lookahead: signal bar {sig_i} >= exec bar {exec_i}")
        if exec_i - sig_i != 1:
            raise AssertionError(f"non-adjacent: exec {exec_i} reads signal {sig_i}")
        if dates is not None and not 0 <= sig_i < len(dates):
            raise AssertionError(f"signal bar {sig_i} outside the panel")


def duplicate_groups(schedules: dict[int, dict[int, int]]) -> dict[str, list[int]]:
    """Group candidate keys by schedule hash: {hash: [D, ...]} for the duplicates table."""
    groups: dict[str, list[int]] = {}
    for key, sch in schedules.items():
        groups.setdefault(schedule_hash(sch), []).append(key)
    return {h: sorted(ks) for h, ks in groups.items()}


def turnover_from_trades(trades: Iterable[dict]) -> float:
    """Total traded notional (both sides) from the engine's audit log, for the cost/turnover
    columns. Uses the executed price, so it includes slippage the same way the ledger does."""
    return float(sum(abs(t["shares"]) * t["price"] for t in trades))
