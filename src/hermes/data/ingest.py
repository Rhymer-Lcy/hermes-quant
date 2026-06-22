"""BaoStock -> adjusted parquet data lake: daily-bar ingestion utilities.

`pull_universe` pulls forward-adjusted daily bars for a list of codes into data/parquet/daily/.
The survivorship-bias fix (point-in-time HS300 membership and the union of all names
ever in the index) lives in hermes.data.membership; the union it produces is the
universe pulled here. There is intentionally no current-membership pull -- that would
reintroduce survivorship bias.
"""
from __future__ import annotations

import pandas as pd

from ..io import atomic_to_parquet
from ..paths import PARQUET_DIR, RAW_DIR, ensure_dirs
from .sources import baostock_source as bss

BACKTEST_START = "2015-01-01"
BACKTEST_END = "2025-12-31"

# Per-board daily price-limit magnitude (price limit, daily ±10%/±20% limit) -- needed
# by the friction-faithful backtest gate (orders at the limit must not fill).
PRICE_LIMIT_PCT = {"Main": 0.10, "STAR": 0.20, "ChiNext": 0.20, "BSE": 0.30}


def board_of(code: str) -> str:
    """Map a BaoStock code ('sh.600000') to its board, for price-limit rules."""
    num = code.split(".")[-1]
    if num.startswith("688"):
        return "STAR"          # STAR Market ±20%
    if num.startswith(("300", "301")):
        return "ChiNext"       # ChiNext ±20%
    if num.startswith(("4", "8", "920")):
        return "BSE"           # Beijing Stock Exchange ±30%
    return "Main"              # Main Board ±10%


def pull_universe(codes, start: str = BACKTEST_START, end: str = BACKTEST_END) -> pd.DataFrame:
    """Pull forward-adjusted daily bars per code -> data/parquet/daily/<code>.parquet.

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
                    atomic_to_parquet(df, out / f"{code.replace('.', '_')}.parquet", index=False)
                results.append({"code": code, "rows": len(df), "status": "ok"})
            except Exception as exc:  # noqa: BLE001 -- record and keep the batch going
                results.append({"code": code, "rows": 0, "status": f"error: {exc}"})
            if i % 25 == 0 or i == n:
                print(f"  ...{i}/{n} pulled")
    return pd.DataFrame(results)


def _is_rebased(old_close: float, new_close: float, rtol: float = 1e-4) -> bool:
    """A forward-adjusted series RE-BASES on a dividend/split: every prior bar's close rescales.
    So if a previously-stored bar's close changes when re-pulled, the series re-based and its whole
    history must be re-pulled; if it is unchanged (within float noise), the new bars are on the same
    basis and can be appended. Errs toward re-pull -- a real re-base shifts the overlap by >> rtol,
    so there are no false negatives (only harmless false positives)."""
    return abs(new_close - old_close) > max(1e-6, abs(old_close) * rtol)


def _latest_published_date(end: str) -> "pd.Timestamp | None":
    """BaoStock's latest available trading date (<= end), via a liquid reference (sh.600000). Used
    only to skip already-current codes on a re-run; returns None (optimization off) on any error."""
    try:
        ref = bss.daily_bars("sh.600000",
                             (pd.Timestamp(end) - pd.Timedelta(days=15)).strftime("%Y-%m-%d"),
                             end, adjustflag="2")
        return pd.to_datetime(ref["date"]).max() if not ref.empty else None
    except Exception:  # noqa: BLE001 -- the reference is an optimization, not a requirement
        return None


def pull_universe_incremental(codes, start: str = BACKTEST_START, end: str = BACKTEST_END) -> pd.DataFrame:
    """Forward-adjusted daily bars, refreshed INCREMENTALLY -- the cheap daily path (the full
    `pull_universe` is for the initial build). Per code: append only the new bars, unless the
    forward-adjusted series re-based (dividend/split, detected via the overlap bar -- see
    `_is_rebased`), in which case that one code is re-pulled in full. Codes already at the latest
    published date are skipped. Same return shape as `pull_universe` (code, rows, status) plus a
    diagnostic `mode`; manages its own session."""
    ensure_dirs()
    out = PARQUET_DIR / "daily"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    n = len(codes)
    with bss.session():
        latest = _latest_published_date(end)
        for i, code in enumerate(codes, 1):
            path = out / f"{code.replace('.', '_')}.parquet"
            try:
                if not path.exists():                                       # new name -> full pull
                    df = bss.daily_bars(code, start, end, adjustflag="2")
                    if not df.empty:
                        atomic_to_parquet(df, path, index=False)
                    results.append({"code": code, "rows": len(df), "status": "ok", "mode": "full:new"})
                    continue
                existing = pd.read_parquet(path)
                ex_dates = pd.to_datetime(existing["date"])
                last = ex_dates.max()
                if latest is not None and last >= latest:                   # already current -> skip
                    results.append({"code": code, "rows": len(existing), "status": "ok", "mode": "skip"})
                    continue
                win = bss.daily_bars(code, last.strftime("%Y-%m-%d"), end, adjustflag="2")
                if win.empty:
                    results.append({"code": code, "rows": len(existing), "status": "ok", "mode": "no-new"})
                    continue
                win_dates = pd.to_datetime(win["date"])
                old_o, new_o = existing.loc[ex_dates == last, "close"], win.loc[win_dates == last, "close"]
                rebased = (len(old_o) == 0 or len(new_o) == 0
                           or _is_rebased(float(old_o.iloc[0]), float(new_o.iloc[0])))
                if rebased:                                                 # dividend/split -> full re-pull
                    df = bss.daily_bars(code, start, end, adjustflag="2")
                    if not df.empty:
                        atomic_to_parquet(df, path, index=False)
                    results.append({"code": code, "rows": len(df), "status": "ok", "mode": "full:rebased"})
                    continue
                add = win.loc[win_dates > last]                             # same basis -> safe append
                if not add.empty:
                    combined = (pd.concat([existing, add], ignore_index=True)
                                .drop_duplicates("date", keep="last").reset_index(drop=True))
                    atomic_to_parquet(combined, path, index=False)
                results.append({"code": code, "rows": len(existing) + len(add), "status": "ok",
                                "mode": f"append:+{len(add)}"})
            except Exception as exc:  # noqa: BLE001 -- record and keep the batch going
                results.append({"code": code, "rows": 0, "status": f"error: {exc}", "mode": "error"})
            if i % 50 == 0 or i == n:
                print(f"  ...{i}/{n} ({sum(r['status'] == 'ok' for r in results)} ok)")
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
