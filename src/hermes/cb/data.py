"""Convertible-bond lake builders -- free sources, survivorship-free.

Feeds the pre-registered double-low study (docs/cb_lake.md). Endpoints validated by
scripts/probes/probe_cb_freedata.py:

  - Eastmoney listing (bond_zh_cov): every bond ever issued -- matured, forced-called and
    defaulted included -- with the underlying-stock code, issue size, listing date.
  - Sina daily bars (bond_zh_hs_cov_daily): whole-life OHLCV for dead and live bonds.
    The tradable price/volume record the study prices from.
  - Eastmoney value analysis (bond_zh_cov_value_analysis): dated close / conversion-value /
    premium series over each bond's whole life (conversion price point-in-time by
    construction). The double-low score input.
  - JSL revision log (bond_cb_adj_logs_jsl): explicit downward-revision records. A
    cross-check input only, never a score input.

All four are scrapers, so pulls are rate-limited, retried with backoff, and RESUMABLE: one
parquet per bond, skip-if-present (delete a file, or data/parquet/cb/ wholesale, to force a
re-pull). AKShare is imported lazily (intraday/data.py precedent) so importing hermes.cb
never requires a network-capable environment.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ..io import atomic_to_parquet
from ..paths import PARQUET_DIR

CB_DIR = PARQUET_DIR / "cb"
UNIVERSE_PARQUET = CB_DIR / "universe.parquet"
BARS_DIR = CB_DIR / "bars"
PREMIUM_DIR = CB_DIR / "premium"
REVISIONS_DIR = CB_DIR / "revisions"

_SLEEP_SEC = 0.35     # polite spacing per worker between scraper calls
_BACKOFF_SEC = 3.0
_MAX_WORKERS = 4      # both endpoints answer in 2-4 s; 4 workers keeps ~2 req/s aggregate

_UNIVERSE_COLS = {
    "债券代码": "code",
    "债券简称": "name",
    "正股代码": "stock_code",
    "正股简称": "stock_name",
    "发行规模": "issue_size",     # 1e8 CNY, at issuance (static; PIT remaining size is not free)
    "上市时间": "listing_date",
    "信用评级": "rating",         # today's snapshot, not point-in-time -- never a filter input
}
_PREMIUM_COLS = {
    "日期": "date",
    "收盘价": "close",
    "纯债价值": "bond_floor",
    "转股价值": "conv_value",
    "纯债溢价率": "bond_premium",  # percent
    "转股溢价率": "conv_premium",  # percent, e.g. 30.5 == 30.5%
}
_REVISION_COLS = {
    "转债名称": "name",
    "股东大会日": "meeting_date",
    "下修前转股价": "old_conv_price",
    "下修后转股价": "new_conv_price",
    "新转股价生效日期": "effective_date",
    "下修底价": "floor_price",
}


def sina_symbol(code: str) -> str:
    """Sina prefixes Shanghai CB codes (11xxxx) with 'sh' and Shenzhen (12xxxx) with 'sz'."""
    return ("sh" if code.startswith("11") else "sz") + code


def _retry(pull: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """One retry, and only for TRANSIENT errors. A KeyError (AKShare parsing an empty
    vendor response) or an empty frame means the source has nothing for this bond --
    retrying that just multiplies request-timeout waits across hundreds of dead codes."""
    try:
        return pull()
    except (KeyError, ValueError):
        raise
    except Exception:
        time.sleep(_BACKOFF_SEC)
        return pull()


def build_universe() -> pd.DataFrame:
    """Pull the Eastmoney listing into data/parquet/cb/universe.parquet and return it.

    Keeps every bond with a listing date. Drops the 40xxxx delisted-segment re-codes (the
    same bond under a new code after its main-board delisting; the probe showed that
    segment's bars are not served anyway -- hence the study's terminal-value convention)."""
    import akshare as ak

    raw = _retry(ak.bond_zh_cov)
    df = raw[list(_UNIVERSE_COLS)].rename(columns=_UNIVERSE_COLS)
    df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")
    df["issue_size"] = pd.to_numeric(df["issue_size"], errors="coerce")
    df = (df[df["listing_date"].notna() & ~df["code"].str.startswith("40")]
          .drop_duplicates("code").sort_values("listing_date").reset_index(drop=True))
    CB_DIR.mkdir(parents=True, exist_ok=True)
    atomic_to_parquet(df, UNIVERSE_PARQUET, index=False)
    return df


def _pull_bars(code: str) -> pd.DataFrame:
    import akshare as ak

    df = ak.bond_zh_hs_cov_daily(symbol=sina_symbol(code))
    df["date"] = pd.to_datetime(df["date"])
    num = ["open", "high", "low", "close", "volume"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def _pull_premium(code: str) -> pd.DataFrame:
    import akshare as ak

    df = ak.bond_zh_cov_value_analysis(symbol=code).rename(columns=_PREMIUM_COLS)
    df["date"] = pd.to_datetime(df["date"])
    num = [c for c in df.columns if c != "date"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def _pull_revisions(code: str) -> pd.DataFrame:
    import akshare as ak

    df = ak.bond_cb_adj_logs_jsl(symbol=code).rename(columns=_REVISION_COLS)
    for col in ("meeting_date", "effective_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("old_conv_price", "new_conv_price", "floor_price"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def _build_per_bond(codes: Iterable[str], out_dir: Path,
                    pull: Callable[[str], pd.DataFrame], what: str,
                    max_workers: int = _MAX_WORKERS, force: bool = False) -> dict[str, str]:
    """Pull `what` for every code into out_dir/<code>.parquet, skipping codes already
    resolved either way (resume): a served bond has a parquet, a miss has a <code>.miss
    marker holding the error, so re-runs never re-burn timeout waits on dead codes.
    Delete the .miss files (scripts/build_cb_lake.py --retry-misses) to try them again.
    `force=True` re-pulls every given code regardless (atomic overwrite; a success clears
    a stale .miss) -- the nightly refresh path. Returns {code: error} for THIS run's
    misses; an empty frame counts as a miss.

    Pulls run on a small thread pool (endpoint latency dominates at 2-4 s per call);
    workers write distinct files, atomically, so a kill mid-run loses nothing."""

    def pull_one(code: str) -> tuple[str, str | None]:
        try:
            df = _retry(lambda: pull(code))
            if df.empty:
                raise ValueError("empty frame")
            atomic_to_parquet(df, out_dir / f"{code}.parquet", index=False)
            (out_dir / f"{code}.miss").unlink(missing_ok=True)
            return code, None
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            (out_dir / f"{code}.miss").write_text(err, encoding="utf-8")
            return code, err
        finally:
            time.sleep(_SLEEP_SEC)

    out_dir.mkdir(parents=True, exist_ok=True)
    todo = list(codes) if force else [c for c in codes if not (out_dir / f"{c}.parquet").exists()
                                      and not (out_dir / f"{c}.miss").exists()]
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool_:
        for i, (code, err) in enumerate(pool_.map(pull_one, todo), 1):
            if err is not None:
                failures[code] = err
            if i % 50 == 0 or i == len(todo):
                print(f"   {what}: {i}/{len(todo)} pulled, {len(failures)} misses", flush=True)
    return failures


def build_bars(codes: Iterable[str]) -> dict[str, str]:
    return _build_per_bond(codes, BARS_DIR, _pull_bars, "bars")


def build_premium(codes: Iterable[str]) -> dict[str, str]:
    return _build_per_bond(codes, PREMIUM_DIR, _pull_premium, "premium")


def build_revisions(codes: Iterable[str]) -> dict[str, str]:
    """JSL logs for the cross-check SAMPLE only (the study script picks the sample). A miss
    here is ambiguous -- 'no revisions ever' and 'not served' look the same -- so misses are
    excluded from the cross-check denominator rather than counted as mismatches. Single
    worker: JSL is a small site, and a rate-limit block would silently shrink that
    denominator."""
    return _build_per_bond(codes, REVISIONS_DIR, _pull_revisions, "revisions", max_workers=1)


def refresh_recent(max_age_days: int = 45) -> dict:
    """Nightly incremental refresh for the forward paper record: re-pull the universe
    listing, then bars and premium for every bond that could still be trading -- last
    served bar within `max_age_days`, or listed (per the listing table) but not served
    yet. Whole-life re-pulls with atomic overwrite (the sources send the full series
    anyway); long-dead bonds are never touched again. Returns
    {'attempted': N, 'bars': misses, 'premium': misses}."""
    uni = build_universe()
    listed = uni.loc[uni["listing_date"] <= pd.Timestamp.now(), "code"]
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=max_age_days)
    todo = []
    for code in listed:
        f = BARS_DIR / f"{code}.parquet"
        if not f.exists():
            todo.append(code)          # never served: a fresh listing may have appeared
        elif pd.read_parquet(f, columns=["date"])["date"].max() >= cutoff:
            todo.append(code)
    return {
        "attempted": len(todo),
        "bars": _build_per_bond(todo, BARS_DIR, _pull_bars, "bars", force=True),
        "premium": _build_per_bond(todo, PREMIUM_DIR, _pull_premium, "premium", force=True),
    }


def load_universe() -> pd.DataFrame:
    return pd.read_parquet(UNIVERSE_PARQUET)


def _load_dir(dir_: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(dir_.glob("*.parquet")):
        df = pd.read_parquet(p)
        df.insert(0, "code", p.stem)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no per-bond parquets under {dir_}; run scripts/build_cb_lake.py")
    return pd.concat(frames, ignore_index=True)


def load_bars() -> pd.DataFrame:
    """Long frame [code, date, open, high, low, close, volume], every bond on disk."""
    return _load_dir(BARS_DIR)


def load_premium() -> pd.DataFrame:
    """Long frame [code, date, close, bond_floor, conv_value, premiums], every bond on disk."""
    return _load_dir(PREMIUM_DIR)


def load_revisions() -> pd.DataFrame:
    """Long frame of the sampled JSL downward-revision logs."""
    return _load_dir(REVISIONS_DIR)
