"""Annual-report fundamentals, CSRC industry buckets, and the SSE margin-balance series.

Built for the pre-registered friend's-ruleset studies (issues #2-#6). Three small lakes:

- ``profit_annual.parquet`` -- one row per (code, fiscal year): full-year ROE (BaoStock
  ``roeAvg``, decimal) plus the actual publication date. Publication-date alignment is the
  whole point: an annual report typically publishes the following March-April, and a quality
  gate that counted it any earlier would leak.
- ``industry_csrc.parquet`` -- the CSRC industry classification snapshot; the leading class
  code (e.g. ``J66``) maps names into the five buckets frozen in issue #4's appendix.
  DISCLOSED in the issues: a current snapshot, not point-in-time.
- ``margin_sse.parquet`` -- SSE daily margin financing balance (the issue #6 sentiment
  series; SSE-only is the frozen market proxy).
"""
from __future__ import annotations

import re
import time

import pandas as pd

from ..paths import PARQUET_DIR

PROFIT_ANNUAL_PARQUET = PARQUET_DIR / "profit_annual.parquet"
INDUSTRY_PARQUET = PARQUET_DIR / "industry_csrc.parquet"
MARGIN_SSE_PARQUET = PARQUET_DIR / "margin_sse.parquet"
DIVIDENDS_PARQUET = PARQUET_DIR / "dividends.parquet"
RAW_CLOSE_PARQUET = PARQUET_DIR / "raw_close.parquet"

# The five-bucket CSRC mapping frozen in issue #4's appendix (shared by issues #2-#6).
BUCKETS = {
    "finance": {"J66"},
    "livelihood": {"D44", "G54"},
    "consumption": {"C36", "C38", "C15"},
    "tech": {"C34", "C35", "C39", "I64", "I65"},
    "infrastructure": {"E47", "E48", "C30"},
}
CYCLICAL_BUCKETS = ("tech", "infrastructure")   # the sleeves the friend treats as cyclicals

_CSRC_CODE = re.compile(r"^[A-Z]\d{2}")


def csrc_class(industry: str) -> str | None:
    """Extract the leading CSRC class code ('J66') from an `industry` string, else None.
    Operates on the ASCII prefix only, so it is immune to the GBK-vs-UTF-8 rendering of the
    Chinese remainder of the string."""
    m = _CSRC_CODE.match(str(industry or ""))
    return m.group(0) if m else None


def bucket_of(industry: str) -> str | None:
    """Map an `industry` string to its frozen five-bucket name, else None (outside the map)."""
    code = csrc_class(industry)
    if code is None:
        return None
    for name, codes in BUCKETS.items():
        if code in codes:
            return name
    return None


def _relogin_patiently(bss, max_tries: int = 6) -> None:
    """Re-establish a dropped BaoStock session, tolerating the case where the RE-LOGIN itself
    hits the transport-error family (observed: a throttling cascade takes out login too). Backs
    off 5s -> 10s -> ... -> 60s; only after `max_tries` straight failures does it give up."""
    for k in range(max_tries):
        try:
            bss.relogin()
            return
        except Exception as exc:  # noqa: BLE001 -- only the transient family is worth waiting on
            if k + 1 >= max_tries or not bss.is_transport_error(str(exc)):
                raise
            time.sleep(min(60.0, 5.0 * 2.0 ** k))


def _pull_done_path():
    from ..paths import RAW_DIR
    return RAW_DIR / "profit_pull_done.txt"


def pull_annual_profit(codes, years, pause: float = 0.2) -> pd.DataFrame:
    """Pull annual-report profitability for every (code, year) -> profit_annual.parquet.

    RESUMABLE: completed codes are checkpointed (data/raw/profit_pull_done.txt plus the
    accumulated parquet) every 50 names, and a re-run skips them -- so a mid-batch throttling
    cascade costs minutes, not the whole pull. A code is marked done only if EVERY year query
    succeeded; failed codes stay eligible for the next run. The pace and the patient re-login
    follow the daily-bar ingest lessons (hammering triggers server-side throttling; a partial
    lake that loads without complaint is how a study silently inverts its verdict, so the
    caller must treat leftover failed codes as a hard stop, not a warning).

    Returns the assembled table; an empty (code, year) -- not yet listed -- is simply absent.
    """
    from ..io import atomic_to_parquet, atomic_write_text
    from .sources import baostock_source as bss

    done_path = _pull_done_path()
    done: set[str] = (set(done_path.read_text(encoding="utf-8").split())
                      if done_path.exists() else set())
    table = (pd.read_parquet(PROFIT_ANNUAL_PARQUET) if PROFIT_ANNUAL_PARQUET.exists()
             else pd.DataFrame(columns=["code", "pubDate", "statDate", "roeAvg"]))
    todo = [c for c in codes if c not in done]
    if done:
        print(f"  resuming: {len(done)} codes already complete, {len(todo)} to go")

    def checkpoint(frames: list[pd.DataFrame]) -> pd.DataFrame:
        merged = pd.concat([table, *frames], ignore_index=True)
        merged = (merged.drop_duplicates(["code", "statDate"], keep="last")
                        .sort_values(["code", "statDate"]).reset_index(drop=True))
        atomic_to_parquet(merged, PROFIT_ANNUAL_PARQUET, index=False)
        atomic_write_text("\n".join(sorted(done)), done_path)
        return merged

    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    n = len(todo)
    with bss.session():
        for i, code in enumerate(todo, 1):
            code_frames, code_ok = [], True
            for year in years:
                for attempt in range(4):
                    try:
                        df = bss.annual_profit(code, year)
                        if not df.empty:
                            code_frames.append(df[["code", "pubDate", "statDate", "roeAvg"]])
                        break
                    except Exception as exc:  # noqa: BLE001 -- retry the recoverable families
                        msg = str(exc)
                        recoverable = bss.is_session_error(msg) or bss.is_transport_error(msg)
                        if attempt + 1 >= 4 or not recoverable:
                            failed.append(f"{code}/{year}: {msg}")
                            code_ok = False
                            break
                        time.sleep(2.0 ** attempt)
                        _relogin_patiently(bss)
                time.sleep(pause)
            if code_ok:
                frames.extend(code_frames)
                done.add(code)
            if i % 50 == 0 or i == n:
                table = checkpoint(frames)
                frames = []
                print(f"  ...{i}/{n} names ({len(failed)} failed queries)")
    table = checkpoint(frames)
    if failed:
        print(f"  {len(failed)} (code, year) queries failed; first few: {failed[:5]}")
        print("  the failed codes are NOT checkpointed -- re-run to retry them")
    return table


def pull_industry_snapshot() -> pd.DataFrame:
    """Pull the current CSRC industry snapshot -> industry_csrc.parquet."""
    from ..io import atomic_to_parquet
    from .sources import baostock_source as bss

    with bss.session():
        df = bss.stock_industry()
    df = df[["updateDate", "code", "industry"]].copy()
    df["csrc"] = df["industry"].map(csrc_class)
    df["bucket"] = df["industry"].map(bucket_of)
    atomic_to_parquet(df, INDUSTRY_PARQUET, index=False)
    return df


def pull_margin_sse(start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:
    """Pull the SSE daily margin financing balance -> margin_sse.parquet.

    AKShare's `stock_margin_sse` wraps the exchange's own disclosure. Pulled in yearly chunks
    for robustness; kept columns: date, rzye (融资余额, RMB).
    """
    import akshare as ak

    from ..io import atomic_to_parquet

    end_ts = pd.Timestamp(end) if end else pd.Timestamp.now().normalize()
    chunks = []
    for y in range(pd.Timestamp(start).year, end_ts.year + 1):
        s = max(pd.Timestamp(start), pd.Timestamp(f"{y}-01-01"))
        e = min(end_ts, pd.Timestamp(f"{y}-12-31"))
        df = ak.stock_margin_sse(start_date=s.strftime("%Y%m%d"), end_date=e.strftime("%Y%m%d"))
        if not df.empty:
            chunks.append(df)
    raw = pd.concat(chunks, ignore_index=True)
    out = pd.DataFrame({
        "date": pd.to_datetime(raw["信用交易日期"], format="%Y%m%d"),
        "rzye": pd.to_numeric(raw["融资余额"], errors="coerce"),
    }).dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)
    atomic_to_parquet(out, MARGIN_SSE_PARQUET, index=False)
    return out


def pull_dividends(codes, years, pause: float = 0.1) -> pd.DataFrame:
    """Cash-dividend events for every (code, ex-year) -> dividends.parquet. Kept columns:
    code, ex_date (除权除息日 -- the point-in-time anchor: a declared dividend counts only
    once it has gone ex), dps (per-share pre-tax cash). Small batches only (a sector, not
    the whole lake); manages its own session."""
    import baostock as bs

    from ..io import atomic_to_parquet
    from .sources import baostock_source as bss

    rows = []
    with bss.session():
        for code in codes:
            for year in years:
                rs = bs.query_dividend_data(code=code, year=str(year), yearType="operate")
                if rs.error_code != "0":
                    raise RuntimeError(f"dividend query failed for {code}/{year}: "
                                       f"{rs.error_code} {rs.error_msg}")
                while rs.next():
                    rows.append(dict(zip(rs.fields, rs.get_row_data())))
                time.sleep(pause)
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "code": df["code"],
        "ex_date": pd.to_datetime(df["dividOperateDate"], errors="coerce"),
        "dps": pd.to_numeric(df["dividCashPsBeforeTax"], errors="coerce"),
    }).dropna(subset=["ex_date"]).fillna({"dps": 0.0})
    out = out.sort_values(["code", "ex_date"]).reset_index(drop=True)
    atomic_to_parquet(out, DIVIDENDS_PARQUET, index=False)
    return out


def pull_raw_close(codes, start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:
    """UNADJUSTED daily closes for a small set of codes -> raw_close.parquet (wide). The
    dividend yield must divide by the price actually quoted that day; the adjusted lake
    series would distort every historical yield."""
    from ..io import atomic_to_parquet
    from .sources import baostock_source as bss

    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    series = {}
    with bss.session():
        for code in codes:
            df = bss.daily_bars(code, start, end, adjustflag="3")
            if not df.empty:
                series[code] = df.set_index("date")["close"]
            time.sleep(0.2)
    panel = pd.DataFrame(series).sort_index()
    atomic_to_parquet(panel, RAW_CLOSE_PARQUET)
    return panel


def trailing_yield(dividends: pd.DataFrame, raw_close: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time trailing dividend yield: the 365-calendar-day rolling sum of per-share
    cash dividends (counted from ex-date) divided by the unadjusted close."""
    # Roll on a FULL calendar reaching 365 days before the price window, so the trailing sum
    # at the window's start still sees the prior year's ex-dates; then align to trading days.
    full = pd.date_range(raw_close.index.min() - pd.Timedelta(days=370), raw_close.index.max())
    out = pd.DataFrame(index=raw_close.index, columns=raw_close.columns, dtype=float)
    for code in raw_close.columns:
        ev = dividends.loc[dividends["code"] == code].groupby("ex_date")["dps"].sum()
        ttm = ev.reindex(full, fill_value=0.0).rolling("365D").sum().reindex(raw_close.index)
        out[code] = ttm / raw_close[code]
    return out


def load_annual_roe() -> pd.DataFrame:
    """The profit_annual lake as a typed table (code, pubDate, statDate, roeAvg)."""
    df = pd.read_parquet(PROFIT_ANNUAL_PARQUET)
    df["pubDate"] = pd.to_datetime(df["pubDate"])
    df["statDate"] = pd.to_datetime(df["statDate"])
    return df


def quality_mask(dates: pd.DatetimeIndex, codes, roe: pd.DataFrame | None = None,
                 n_reports: int = 3, threshold: float = 0.15) -> pd.DataFrame:
    """Point-in-time quality gate (issue #2): True at (date, code) when the `n_reports` most
    recently PUBLISHED annual reports each show ROE > `threshold`.

    A report counts only from its `pubDate`. A name with fewer than `n_reports` published
    reports is not eligible (False), rather than being judged on a thinner window.
    """
    roe = load_annual_roe() if roe is None else roe
    out = pd.DataFrame(False, index=dates, columns=list(codes))
    for code, grp in roe.groupby("code"):
        if code not in out.columns:
            continue
        g = grp.sort_values("pubDate")
        # As of each publication, the rolling min over the last n published annual ROEs.
        ok = (g["roeAvg"].rolling(n_reports).min() > threshold).to_numpy()
        step = pd.Series(ok, index=g["pubDate"].to_numpy())
        step = step[~step.index.duplicated(keep="last")]
        aligned = step.reindex(dates, method="ffill").eq(True)   # NaN before first pub -> False
        out[code] = aligned.to_numpy(dtype=bool)
    return out
