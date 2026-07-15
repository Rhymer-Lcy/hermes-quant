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


def pull_annual_profit(codes, years, pause: float = 0.1) -> pd.DataFrame:
    """Pull annual-report profitability for every (code, year) -> profit_annual.parquet.

    Manages its own BaoStock session with the same drop/transport recovery discipline as the
    daily-bar ingest (a dropped session poisons every subsequent query in a serial batch).
    Returns the assembled table; an empty (code, year) -- not yet listed -- is simply absent.
    """
    from ..io import atomic_to_parquet
    from .sources import baostock_source as bss

    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    n = len(codes)
    with bss.session():
        for i, code in enumerate(codes, 1):
            for year in years:
                for attempt in range(4):
                    try:
                        df = bss.annual_profit(code, year)
                        if not df.empty:
                            frames.append(df[["code", "pubDate", "statDate", "roeAvg"]])
                        break
                    except Exception as exc:  # noqa: BLE001 -- retry the recoverable families
                        msg = str(exc)
                        recoverable = bss.is_session_error(msg) or bss.is_transport_error(msg)
                        if attempt + 1 >= 4 or not recoverable:
                            errors.append(f"{code}/{year}: {msg}")
                            break
                        time.sleep(2.0 ** attempt)
                        bss.relogin()
                time.sleep(pause)
            if i % 25 == 0 or i == n:
                print(f"  ...{i}/{n} names ({len(errors)} errors)")
    table = (pd.concat(frames, ignore_index=True) if frames
             else pd.DataFrame(columns=["code", "pubDate", "statDate", "roeAvg"]))
    table = table.sort_values(["code", "statDate"]).reset_index(drop=True)
    atomic_to_parquet(table, PROFIT_ANNUAL_PARQUET, index=False)
    if errors:
        print(f"  {len(errors)} (code, year) queries failed; first few: {errors[:5]}")
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
        aligned = step.reindex(dates, method="ffill").fillna(False).astype(bool)
        out[code] = aligned.to_numpy()
    return out
