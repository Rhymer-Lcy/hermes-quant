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
HOLDER_COUNTS_PARQUET = PARQUET_DIR / "holder_counts.parquet"

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


def _done_path(name: str):
    """Per-pull resume checkpoint (a plain list of completed codes) under data/raw/."""
    from ..paths import RAW_DIR
    return RAW_DIR / f"{name}_pull_done.txt"


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
        objs = [f for f in [table, *frames] if not f.empty]
        merged = pd.concat(objs, ignore_index=True) if objs else table
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


def _dividend_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Typed (code, ex_date, dps) rows from raw BaoStock dividend records."""
    return pd.DataFrame({
        "code": raw["code"],
        "ex_date": pd.to_datetime(raw["dividOperateDate"], errors="coerce"),
        "dps": pd.to_numeric(raw["dividCashPsBeforeTax"], errors="coerce"),
    }).dropna(subset=["ex_date"]).fillna({"dps": 0.0})


def pull_dividends(codes, years, pause: float = 0.1) -> pd.DataFrame:
    """Cash-dividend events for every (code, ex-year) -> dividends.parquet. Kept columns:
    code, ex_date (除权除息日 -- the point-in-time anchor: a declared dividend counts only
    once it has gone ex), dps (per-share pre-tax cash).

    RESUMABLE like `pull_annual_profit`: completed codes checkpoint to
    data/raw/dividend_pull_done.txt every 50 names; a code counts as done only if EVERY
    year query succeeded, so failed codes stay eligible for the next run."""
    import baostock as bs

    from ..io import atomic_to_parquet, atomic_write_text
    from .sources import baostock_source as bss

    done_path = _done_path("dividend")
    done: set[str] = (set(done_path.read_text(encoding="utf-8").split())
                      if done_path.exists() else set())
    table = (pd.read_parquet(DIVIDENDS_PARQUET) if DIVIDENDS_PARQUET.exists()
             else pd.DataFrame(columns=["code", "ex_date", "dps"]))
    todo = [c for c in codes if c not in done]
    if done:
        print(f"  resuming: {len(done)} codes already complete, {len(todo)} to go")

    def checkpoint(frames: list[pd.DataFrame]) -> pd.DataFrame:
        objs = [f for f in [table, *frames] if not f.empty]
        merged = pd.concat(objs, ignore_index=True) if objs else table
        merged = (merged.drop_duplicates().sort_values(["code", "ex_date"])
                        .reset_index(drop=True))
        atomic_to_parquet(merged, DIVIDENDS_PARQUET, index=False)
        atomic_write_text("\n".join(sorted(done)), done_path)
        return merged

    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    n = len(todo)
    with bss.session():
        for i, code in enumerate(todo, 1):
            code_rows, code_ok = [], True
            for year in years:
                for attempt in range(4):
                    try:
                        rs = bs.query_dividend_data(code=code, year=str(year),
                                                    yearType="operate")
                        if rs.error_code != "0":
                            raise RuntimeError(f"{rs.error_code} {rs.error_msg}")
                        while rs.next():
                            code_rows.append(dict(zip(rs.fields, rs.get_row_data())))
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
                if code_rows:
                    frames.append(_dividend_rows(pd.DataFrame(code_rows)))
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


def pull_raw_close(codes, start: str = "2015-01-01", end: str | None = None,
                   pause: float = 0.2) -> pd.DataFrame:
    """UNADJUSTED daily closes -> raw_close.parquet (wide). The dividend yield must divide
    by the price actually quoted that day; the adjusted lake series would distort every
    historical yield.

    RESUMABLE: completed codes checkpoint to data/raw/raw_close_pull_done.txt every 25
    names; a code with no bars in the window still counts as pulled (in the done file,
    absent from the panel). Codes already present as panel columns are skipped."""
    from ..io import atomic_to_parquet, atomic_write_text
    from .sources import baostock_source as bss

    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    done_path = _done_path("raw_close")
    done: set[str] = (set(done_path.read_text(encoding="utf-8").split())
                      if done_path.exists() else set())
    panel = pd.read_parquet(RAW_CLOSE_PARQUET) if RAW_CLOSE_PARQUET.exists() else pd.DataFrame()
    done |= set(panel.columns)
    todo = [c for c in codes if c not in done]
    if done:
        print(f"  resuming: {len(done)} codes already complete, {len(todo)} to go")

    def checkpoint(series: dict) -> pd.DataFrame:
        merged = pd.concat([panel, pd.DataFrame(series)], axis=1).sort_index()
        atomic_to_parquet(merged, RAW_CLOSE_PARQUET)
        atomic_write_text("\n".join(sorted(done)), done_path)
        return merged

    series: dict = {}
    failed: list[str] = []
    n = len(todo)
    with bss.session():
        for i, code in enumerate(todo, 1):
            for attempt in range(4):
                try:
                    df = bss.daily_bars(code, start, end, adjustflag="3")
                    if not df.empty:
                        series[code] = df.set_index("date")["close"]
                    done.add(code)
                    break
                except Exception as exc:  # noqa: BLE001 -- retry the recoverable families
                    msg = str(exc)
                    recoverable = bss.is_session_error(msg) or bss.is_transport_error(msg)
                    if attempt + 1 >= 4 or not recoverable:
                        failed.append(f"{code}: {msg}")
                        break
                    time.sleep(2.0 ** attempt)
                    _relogin_patiently(bss)
            time.sleep(pause)
            if i % 25 == 0 or i == n:
                panel = checkpoint(series)
                series = {}
                print(f"  ...{i}/{n} names ({len(failed)} failed)")
    panel = checkpoint(series)
    if failed:
        print(f"  {len(failed)} codes failed; first few: {failed[:5]} -- re-run to retry")
    return panel


def _holder_counts_f10(code: str) -> pd.DataFrame | None:
    """Fallback for names absent from the akshare DET report (observed for delisted names
    and a minority of live ones): the same vendor's F10 holder-number report
    (RPT_F10_EH_HOLDERNUM), same fields incl. the NOTICE_DATE point-in-time anchor.
    Returns None when the vendor has no data for the code at all."""
    import requests

    exch, num = code.split(".")
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    rows, page = [], 1
    while True:
        params = {"sortColumns": "END_DATE", "sortTypes": "-1", "pageSize": "500",
                  "pageNumber": str(page), "reportName": "RPT_F10_EH_HOLDERNUM",
                  "columns": "SECUCODE,END_DATE,NOTICE_DATE,HOLDER_TOTAL_NUM,"
                             "HOLDER_TOTAL_NUMCHANGE",
                  "filter": f'(SECUCODE="{num}.{exch.upper()}")',
                  "source": "WEB", "client": "WEB"}
        res = requests.get(url, params=params, timeout=30).json().get("result")
        if res is None:
            return None if page == 1 else pd.DataFrame(rows)
        rows.extend(res["data"])
        if page >= res["pages"]:
            break
        page += 1
    raw = pd.DataFrame(rows)
    holders = pd.to_numeric(raw["HOLDER_TOTAL_NUM"], errors="coerce")
    change = pd.to_numeric(raw["HOLDER_TOTAL_NUMCHANGE"], errors="coerce")
    return pd.DataFrame({
        "code": code,
        "stat_date": pd.to_datetime(raw["END_DATE"], errors="coerce"),
        "pub_date": pd.to_datetime(raw["NOTICE_DATE"], errors="coerce"),
        "holders": holders,
        "prior_holders": holders - change,
    }).dropna(subset=["stat_date", "pub_date", "holders"])


def pull_holder_counts(codes, pause: float = 0.5) -> pd.DataFrame:
    """Shareholder-count disclosure history per name (Eastmoney via akshare
    `stock_zh_a_gdhs_detail_em`, falling back to the vendor's F10 report for names the DET
    report lacks) -> holder_counts.parquet. Kept columns: code, stat_date (统计截止日),
    pub_date (公告日期 -- the point-in-time anchor: the market learns the count only at
    publication), holders (本次), prior_holders (上次).

    RESUMABLE: completed codes checkpoint to data/raw/holder_pull_done.txt every 25 names;
    an empty vendor response still counts as pulled."""
    import akshare as ak

    from ..io import atomic_to_parquet, atomic_write_text

    done_path = _done_path("holder")
    done: set[str] = (set(done_path.read_text(encoding="utf-8").split())
                      if done_path.exists() else set())
    table = (pd.read_parquet(HOLDER_COUNTS_PARQUET) if HOLDER_COUNTS_PARQUET.exists()
             else pd.DataFrame(columns=["code", "stat_date", "pub_date",
                                        "holders", "prior_holders"]))
    todo = [c for c in codes if c not in done]
    if done:
        print(f"  resuming: {len(done)} codes already complete, {len(todo)} to go")

    def checkpoint(frames: list[pd.DataFrame]) -> pd.DataFrame:
        objs = [f for f in [table, *frames] if not f.empty]
        merged = pd.concat(objs, ignore_index=True) if objs else table
        merged = (merged.drop_duplicates(["code", "stat_date"], keep="last")
                        .sort_values(["code", "stat_date"]).reset_index(drop=True))
        atomic_to_parquet(merged, HOLDER_COUNTS_PARQUET, index=False)
        atomic_write_text("\n".join(sorted(done)), done_path)
        return merged

    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    n = len(todo)
    for i, code in enumerate(todo, 1):
        for attempt in range(4):
            try:
                try:
                    raw = ak.stock_zh_a_gdhs_detail_em(symbol=code.split(".")[1])
                except TypeError:
                    # the DET report has no rows for this name -- try the F10 fallback;
                    # None there too means the vendor has nothing (counts as pulled)
                    f10 = _holder_counts_f10(code)
                    if f10 is not None and not f10.empty:
                        frames.append(f10)
                    done.add(code)
                    break
                if not raw.empty:
                    frames.append(pd.DataFrame({
                        "code": code,
                        "stat_date": pd.to_datetime(raw["股东户数统计截止日"],
                                                    errors="coerce"),
                        "pub_date": pd.to_datetime(raw["股东户数公告日期"], errors="coerce"),
                        "holders": pd.to_numeric(raw["股东户数-本次"], errors="coerce"),
                        "prior_holders": pd.to_numeric(raw["股东户数-上次"], errors="coerce"),
                    }).dropna(subset=["stat_date", "pub_date", "holders"]))
                done.add(code)
                break
            except Exception as exc:  # noqa: BLE001 -- vendor hiccups are worth a few retries
                if attempt + 1 >= 4:
                    failed.append(f"{code}: {exc}")
                    break
                time.sleep(5.0 * 2.0 ** attempt)
        time.sleep(pause)
        if i % 25 == 0 or i == n:
            table = checkpoint(frames)
            frames = []
            print(f"  ...{i}/{n} names ({len(failed)} failed)")
    table = checkpoint(frames)
    if failed:
        print(f"  {len(failed)} codes failed; first few: {failed[:5]} -- re-run to retry")
    return table


def load_holder_counts() -> pd.DataFrame:
    """The holder_counts lake as a typed table (code, stat_date, pub_date, holders,
    prior_holders)."""
    df = pd.read_parquet(HOLDER_COUNTS_PARQUET)
    df["stat_date"] = pd.to_datetime(df["stat_date"])
    df["pub_date"] = pd.to_datetime(df["pub_date"])
    return df


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
