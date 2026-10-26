"""BaoStock -> adjusted parquet data lake: daily-bar ingestion utilities.

`pull_universe` pulls forward-adjusted daily bars for a list of codes into data/parquet/daily/.
The survivorship-bias fix (point-in-time HS300 membership and the union of all names
ever in the index) lives in hermes.data.membership; the union it produces is the
universe pulled here. There is intentionally no current-membership pull -- that would
reintroduce survivorship bias.
"""
from __future__ import annotations

import time

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
                df = _daily_bars_reloginning(code, start, end)
                if not df.empty:
                    write_full_series(df, out / f"{code.replace('.', '_')}.parquet")
                results.append({"code": code, "rows": len(df), "status": "ok"})
            except Exception as exc:  # noqa: BLE001 -- record and keep the batch going
                results.append({"code": code, "rows": 0, "status": f"error: {exc}"})
            if i % 25 == 0 or i == n:
                print(f"  ...{i}/{n} pulled")
            time.sleep(0.2)          # pace the batch: hammering triggers server-side throttling
                                     # (observed as a 1000200x cascade, eventually at login itself)
    return pd.DataFrame(results)


class LakeWriteRefused(RuntimeError):
    """A full-series write would have dropped bars the lake already holds."""


def write_full_series(df: pd.DataFrame, path) -> None:
    """Atomically replace one code's file with a freshly pulled FULL series, but only if the
    replacement still contains every date the stored file holds. A pull with an earlier `end` (the
    research builders default to BACKTEST_END = 2025-12-31), a later `start`, or a vendor gap would
    otherwise overwrite a current series with a shorter one. Raises LakeWriteRefused and writes
    nothing in that case; the batch loops record it as a per-code error."""
    if path.exists():
        stored = pd.to_datetime(pd.read_parquet(path, columns=["date"])["date"])
        lost = sorted(set(stored) - set(pd.to_datetime(df["date"])))
        if lost:
            raise LakeWriteRefused(
                f"refused: the pulled series lacks {len(lost)} stored bar(s) "
                f"({lost[0].date()} .. {lost[-1].date()}); stored series ends {stored.max().date()}, "
                f"pulled series ends {pd.to_datetime(df['date']).max().date()}")
    atomic_to_parquet(df, path, index=False)


def _daily_bars_reloginning(code: str, start: str, end: str, attempts: int = 4) -> pd.DataFrame:
    """daily_bars, recovering from a degraded connection by re-logging in and retrying.

    Two failure families surface in large serial batches, and both poison every SUBSEQUENT query
    until the connection is re-established: a server-side session drop ("10001001 用户未登录")
    and the transport-error family ("1000200x", e.g. 网络接收错误 -- observed to fail 800 straight
    names once it starts). Without recovery the batch runner records the cascade as per-name
    errors while still exiting 0 (a surviving CSI500 pull summary shows one such run: 28 ok of
    886). The proven danger: a study run against the resulting PARTIAL lake does not merely
    degrade -- at 35% coverage the A6 backtest INVERTED its verdict (CSI500 value Calmar printed
    0.32 against the true 0.10). Re-login (which also re-pins the server IP and rebuilds the
    socket) with a growing pause, then retry."""
    for attempt in range(attempts):
        try:
            return bss.daily_bars(code, start, end, adjustflag="2")
        except Exception as exc:  # noqa: BLE001 -- retry only the known recoverable families
            msg = str(exc)
            recoverable = bss.is_session_error(msg) or bss.is_transport_error(msg)
            if attempt + 1 >= attempts or not recoverable:
                raise
            time.sleep(2.0 ** attempt)                    # 1s, 2s, 4s
            bss.relogin()
    raise AssertionError("unreachable")


def _is_rebased(old_close: float, new_close: float, rtol: float = 1e-4) -> bool:
    """A forward-adjusted series RE-BASES on a dividend/split: every prior bar's close rescales.
    So if a previously-stored bar's close changes when re-pulled, the series re-based and its whole
    history must be re-pulled; if it is unchanged (within float noise), the new bars are on the same
    basis and can be appended. Errs toward re-pull -- a real re-base shifts the overlap by >> rtol,
    so there are no false negatives (only harmless false positives)."""
    return abs(new_close - old_close) > max(1e-6, abs(old_close) * rtol)


REBASE_WINDOW_BARS = 60   # stored bars re-checked per code on every incremental refresh


def _window_rebased(stored: pd.DataFrame, fresh: pd.DataFrame, since: pd.Timestamp) -> bool:
    """True if any stored bar dated `since` or later differs from the freshly pulled `fresh` window
    (both frames carry datetime `date` and `close`), or is no longer served at all.

    One overlap bar is not enough. BaoStock can publish an ex-date bar before it applies that
    dividend's factor to the earlier history; the next refresh then compares a bar on or after the
    ex-date, which the factor never touches, and appends onto a stale basis for good. The
    2026-10-24 audit found 346 of 643 HS300-union series in that state, each dividend recorded as a
    price drop. Re-checking the last REBASE_WINDOW_BARS stored bars catches a factor that lands up
    to REBASE_WINDOW_BARS - 2 trading days after its ex-date bar was stored (the window must still
    hold the bar before the ex-date). Two NaN closes count as equal; NaN against a number does not."""
    s = stored.loc[stored["date"] >= since].drop_duplicates("date").set_index("date")["close"]
    f = fresh.drop_duplicates("date").set_index("date")["close"]
    if not s.index.isin(f.index).all():
        return True
    for old, new in zip(s.to_numpy(dtype=float), f.reindex(s.index).to_numpy(dtype=float)):
        if pd.isna(old) and pd.isna(new):
            continue
        if pd.isna(old) or pd.isna(new) or _is_rebased(old, new):
            return True
    return False


def pull_universe_incremental(codes, start: str = BACKTEST_START, end: str = BACKTEST_END) -> pd.DataFrame:
    """Forward-adjusted daily bars, refreshed INCREMENTALLY -- the cheap daily path (the full
    `pull_universe` is for the initial build). Per code: re-pull a window starting at the
    REBASE_WINDOW_BARS-th last stored bar and append only the new bars, unless the forward-adjusted
    series re-based (dividend/split: any stored bar in the window changed -- see `_window_rebased`),
    in which case that one code is re-pulled in full. Every code is checked on every run: there is no
    "already current" shortcut, because the one it replaced took the latest date from a single
    reference stock (a stale or suspended reference silently froze the whole universe) and also hid a
    late factor from a same-day re-run. Same return shape as `pull_universe` (code, rows, status)
    plus a diagnostic `mode`; manages its own session."""
    ensure_dirs()
    out = PARQUET_DIR / "daily"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    n = len(codes)
    with bss.session():
        for i, code in enumerate(codes, 1):
            path = out / f"{code.replace('.', '_')}.parquet"
            try:
                if not path.exists():                                       # new name -> full pull
                    df = _daily_bars_reloginning(code, start, end)
                    if not df.empty:
                        atomic_to_parquet(df, path, index=False)
                    results.append({"code": code, "rows": len(df), "status": "ok", "mode": "full:new"})
                    continue
                existing = pd.read_parquet(path)
                ex_dates = pd.to_datetime(existing["date"])
                last = ex_dates.max()
                since = ex_dates.sort_values().iloc[-min(REBASE_WINDOW_BARS, len(ex_dates))]
                win = _daily_bars_reloginning(code, since.strftime("%Y-%m-%d"), end)
                if win.empty:
                    results.append({"code": code, "rows": len(existing), "status": "ok", "mode": "no-new"})
                    continue
                win_dates = pd.to_datetime(win["date"])
                in_range = ex_dates <= pd.Timestamp(end)      # bars after `end` are out of this pull's scope
                rebased = _window_rebased(pd.DataFrame({"date": ex_dates[in_range],
                                                        "close": existing["close"][in_range]}),
                                          pd.DataFrame({"date": win_dates, "close": win["close"]}),
                                          since)
                if rebased:                                                 # dividend/split -> full re-pull
                    df = _daily_bars_reloginning(code, start, end)
                    if not df.empty:
                        write_full_series(df, path)
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
