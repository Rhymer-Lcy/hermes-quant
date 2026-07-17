"""Build the dividend, unadjusted-close, and shareholder-count lakes (issues #12-#13).

Extends the #9 bank-only dividend and raw-close lakes to the full PIT HS300+CSI500
membership union, and pulls every name's shareholder-count disclosure history from
Eastmoney. All three pulls are resumable: re-running skips completed codes and retries
only failures. Each stage ends with a hard completeness gate (the A6 lesson: a study run
against a silently partial lake is how a verdict inverts).

    conda activate hermes
    python scripts/build_dividend_holder_lakes.py
"""
from __future__ import annotations

import pandas as pd

from hermes.data.fundamentals import (_done_path, pull_dividends, pull_holder_counts,
                                      pull_raw_close)
from hermes.data.membership import CSI500_MEMBERSHIP_PARQUET, MEMBERSHIP_PARQUET
from hermes.paths import ensure_dirs

FIRST_OPERATE_YEAR = 2013   # ex-dates from 2014 feed the trailing yield at the 2015 start


def _gate(name: str, codes: list[str]) -> None:
    done = set(_done_path(name).read_text(encoding="utf-8").split())
    missing = [c for c in codes if c not in done]
    if missing:
        raise SystemExit(f"ABORT: {name}: {len(missing)} codes incomplete "
                         f"(first: {missing[:5]}); re-run to resume")


def main() -> None:
    ensure_dirs()
    hs = pd.read_parquet(MEMBERSHIP_PARQUET)
    cs = pd.read_parquet(CSI500_MEMBERSHIP_PARQUET)
    codes = sorted(set(hs["code"]) | set(cs["code"]))
    years = range(FIRST_OPERATE_YEAR, pd.Timestamp.now().year + 1)
    print(f"union: {len(codes)} names; dividends for operate years "
          f"{years.start}-{years.stop - 1}")

    div = pull_dividends(codes, years)
    print(f"dividend lake: {len(div)} events, {div['code'].nunique()} payers")
    _gate("dividend", codes)

    px = pull_raw_close(codes)
    print(f"raw-close panel: {px.shape[0]} days x {px.shape[1]} names")
    _gate("raw_close", codes)

    hc = pull_holder_counts(codes)
    print(f"holder-count lake: {len(hc)} disclosures, {hc['code'].nunique()} names")
    _gate("holder", codes)


if __name__ == "__main__":
    main()
