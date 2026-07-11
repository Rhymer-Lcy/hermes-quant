"""The CB double-low FORWARD paper record -- the study engine run from a frozen inception.

docs/cb_lake.md froze the design and recorded the study verdict (SURVIVED, pre-registered
criteria); this module starts the only validation left: a forward record. The signal
construction is imported from cb.signals -- the SAME code the study ran -- and the
accounting is the SAME double_low_backtest engine, so the paper record cannot drift from
the research (the live/strategy.py principle). Everything is recompute-from-inception
(idempotent, like the plutus forward record): each evening rebuilds the whole curve, so a
missed run loses nothing.

Frozen at inception, before any forward bar existed:
  - CB_PAPER_INCEPTION 2026-07-10 -- the last close before the first unknown forward day;
    the seed enters the then-current top-20 at the NEXT trading day's close;
  - signals at every later month-end close, execution at the next trading day's close
    (an in-progress month's provisional "month-end" has no next bar yet, so the engine
    inherently waits until the true month-end is confirmed);
  - top-20 equal weight, 0.05% per side (the study's primary cost);
  - turnover floors FROZEN at the study's calibrated values below -- forward eligibility
    must not recalibrate itself as new data arrives;
  - a bond that stops trading exits at its last close (the study's primary mark).
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd

from ..io import atomic_to_parquet, atomic_write_text
from ..paths import PAPER_DIR, ensure_dirs
from . import data as cbdata
from . import signals as sig
from .backtest import double_low_backtest

CB_PAPER_INCEPTION = "2026-07-10"
N_HOLD = 20
COST_PER_SIDE = 5e-4
FROZEN_FLOORS = {"11": 5_109_363.0, "12": 4_867_041.0}   # study calibration, 2026-07-11


def paper_step(as_of: str | None = None, *, persist: bool = True) -> dict:
    """Recompute the forward record from inception on the current lake; return the daily
    report. Idempotent -- safe to re-run, re-date, or skip an evening."""
    bars, prem = cbdata.load_bars(), cbdata.load_premium()
    codes = sorted(set(bars["code"]) & set(prem["code"]))
    p = sig.panels(bars[bars["code"].isin(codes)], prem[prem["code"].isin(codes)])
    if as_of is not None:
        cut = pd.Timestamp(as_of)
        p = {k: v.loc[v.index <= cut] for k, v in p.items()}
    close = p["close"]

    incept = pd.Timestamp(CB_PAPER_INCEPTION)
    if incept not in close.index:
        raise RuntimeError(f"inception {CB_PAPER_INCEPTION} is not a lake trading day")

    base, score_all = sig.base_and_score(close, p["em_close"], p["em_premium"])
    eligible = sig.apply_floor(base, sig.turnover_metric(close, p["volume"]), FROZEN_FLOORS)
    rows = pd.DatetimeIndex(
        sorted({incept} | set(sig.month_end_signals(close.index, CB_PAPER_INCEPTION))))
    score = score_all.where(eligible).loc[rows]

    fwd_close = close.loc[incept:]
    try:
        res = double_low_backtest(fwd_close, score, N_HOLD, COST_PER_SIDE)
        equity = res.equity
    except ValueError:                       # no bar after inception yet: seeded, flat 1.0
        res = None
        equity = pd.Series([1.0], index=[incept], name="equity")

    # The current book = the selection of the last EXECUTED signal (one with a later bar).
    executed = [d for d in score.index if fwd_close.index.get_loc(d) + 1 < len(fwd_close.index)]
    uni = cbdata.load_universe()
    names = dict(zip(uni["code"], uni["name"]))
    book = prev = []
    if executed:
        book = list(score.loc[executed[-1]].dropna().nsmallest(N_HOLD).index)
        if len(executed) > 1:
            prev = list(score.loc[executed[-2]].dropna().nsmallest(N_HOLD).index)
    exec_day = (fwd_close.index[fwd_close.index.get_loc(executed[-1]) + 1]
                if executed else None)

    today = close.index.max()
    run_dt = date.today()
    lag = (run_dt - today.date()).days
    report = {
        "as_of": today.strftime("%Y-%m-%d"),
        "run_date": run_dt.strftime("%Y-%m-%d"),
        "lake_lag_days": lag,
        "fresh": lag <= 4,
        "inception": CB_PAPER_INCEPTION,
        "equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
        "signal_date": executed[-1].strftime("%Y-%m-%d") if executed else None,
        "n_positions": len(book),
        "positions": {c: names.get(c, "") for c in sorted(book)},
        "rebalanced_today": bool(exec_day is not None and exec_day == today),
        "entered_today": sorted(set(book) - set(prev)) if exec_day == today else [],
        "exited_today": sorted(set(prev) - set(book)) if exec_day == today else [],
        "n_rebalances": len(res.rebalances) if res is not None else 0,
    }
    if persist:
        ensure_dirs()
        atomic_to_parquet(equity.to_frame(), PAPER_DIR / "cb_curve.parquet")
        atomic_write_text(json.dumps(report, ensure_ascii=False, indent=2),
                          PAPER_DIR / "cb_report.json")
    return report
