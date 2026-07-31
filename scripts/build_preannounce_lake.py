"""Earnings-preannouncement (业绩预告) lake -> data/parquet/preannounce.parquet. Issue #19.

China mandates a preannouncement whenever a result crosses disclosure thresholds, so the vendor
feed is a scheduled, cross-sectionally comparable disclosure event. Pulled per REPORTING PERIOD
(the vendor's `date=` argument) for every quarter-end 2015-03-31 .. 2025-12-31 and concatenated.

POINT-IN-TIME WARNING, and why this file stores the raw pull. The `date=` argument is the period,
NOT the announcement date: a row for FY2024 can carry 公告日期 in 2025. Any panel built from this
lake must be assembled BY 公告日期 -- the study does that, the lake deliberately does not
pre-filter, so that the raw vendor state stays inspectable.

    python scripts/build_preannounce_lake.py            # full build
    python scripts/build_preannounce_lake.py --resume   # skip periods already in the parquet
"""
from __future__ import annotations

import argparse
import time

import akshare as ak
import pandas as pd

from hermes.io import atomic_to_parquet
from hermes.paths import PARQUET_DIR, ensure_dirs

OUT = PARQUET_DIR / "preannounce.parquet"
START_YEAR, END_YEAR = 2015, 2025
QUARTER_ENDS = ["0331", "0630", "0930", "1231"]


def periods() -> list[str]:
    return [f"{y}{q}" for y in range(START_YEAR, END_YEAR + 1) for q in QUARTER_ENDS]


def pull_period(period: str, attempts: int = 3) -> pd.DataFrame:
    """One reporting period. Empty frame on persistent failure -- the batch keeps going and the
    summary records it, rather than losing a completed multi-hour build to one bad period."""
    for attempt in range(attempts):
        try:
            df = ak.stock_yjyg_em(date=period)
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.copy()
            df["报告期"] = period
            return df
        except Exception as exc:  # noqa: BLE001 -- vendor flakiness is expected in a long batch
            if attempt + 1 >= attempts:
                print(f"  {period}: FAILED after {attempts} attempts ({exc})")
                return pd.DataFrame()
            time.sleep(3.0 * (attempt + 1))
    return pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume", action="store_true", help="skip periods already present in the lake")
    args = ap.parse_args()
    ensure_dirs()

    existing, done = pd.DataFrame(), set()
    if args.resume and OUT.exists():
        existing = pd.read_parquet(OUT)
        done = set(existing["报告期"].astype(str))
        print(f"resuming: {len(done)} periods already in {OUT}")

    todo = [p for p in periods() if p not in done]
    frames = [existing] if not existing.empty else []
    for i, period in enumerate(todo, 1):
        df = pull_period(period)
        if not df.empty:
            frames.append(df)
        print(f"  [{i}/{len(todo)}] {period}: {len(df)} rows")
        if frames:                                   # checkpoint every period; a long pull that
            combined = pd.concat(frames, ignore_index=True)   # dies mid-way loses nothing
            atomic_to_parquet(combined, OUT, index=False)
        time.sleep(0.5)

    final = pd.read_parquet(OUT)
    print(f"\nlake: {len(final):,} rows, {final['报告期'].nunique()} periods -> {OUT}")


if __name__ == "__main__":
    main()
