"""Build the fundamentals lakes for the friend's-ruleset studies (issues #2-#6).

Pulls, in order: the CSRC industry snapshot, the SSE margin-balance series, and annual-report
ROE for every name in the daily lake (fiscal years 2011 onward -- the quality gate needs three
published annual reports before the 2015 backtest start). The ROE pull is ~23k serial BaoStock
queries and takes on the order of an hour; the two small pulls run first so a mid-run failure
of the long one loses nothing else.

    conda activate hermes
    python scripts/build_fundamentals_lake.py
"""
from __future__ import annotations

import pandas as pd

from hermes.data.fundamentals import (pull_annual_profit, pull_industry_snapshot,
                                      pull_margin_sse)
from hermes.paths import PARQUET_DIR, ensure_dirs

FIRST_FISCAL_YEAR = 2011


def main() -> None:
    ensure_dirs()

    ind = pull_industry_snapshot()
    print(f"industry snapshot: {len(ind)} names, "
          f"{ind['bucket'].notna().sum()} inside the five frozen buckets")

    margin = pull_margin_sse()
    print(f"SSE margin balance: {len(margin)} days, "
          f"{margin['date'].min().date()} -> {margin['date'].max().date()}")

    codes = sorted(p.stem.replace("_", ".", 1) for p in (PARQUET_DIR / "daily").glob("*.parquet"))
    years = range(FIRST_FISCAL_YEAR, pd.Timestamp.now().year)
    print(f"annual ROE: {len(codes)} names x fiscal {years.start}-{years.stop - 1} ...")
    roe = pull_annual_profit(codes, years)
    print(f"annual ROE table: {len(roe)} reports, {roe['code'].nunique()} names")


if __name__ == "__main__":
    main()
