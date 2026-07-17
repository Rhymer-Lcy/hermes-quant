"""Build the annual-capex lake (issue #14).

Pulls every union name's annual cash-flow capex line from Eastmoney (NOTICE_DATE is the
point-in-time anchor). Resumable; ends with the hard completeness gate.

    conda activate hermes
    python scripts/build_capex_lake.py
"""
from __future__ import annotations

import pandas as pd

from hermes.data.fundamentals import _done_path, pull_capex
from hermes.data.membership import CSI500_MEMBERSHIP_PARQUET, MEMBERSHIP_PARQUET
from hermes.paths import ensure_dirs


def main() -> None:
    ensure_dirs()
    hs = pd.read_parquet(MEMBERSHIP_PARQUET)
    cs = pd.read_parquet(CSI500_MEMBERSHIP_PARQUET)
    codes = sorted(set(hs["code"]) | set(cs["code"]))
    print(f"union: {len(codes)} names")

    cap = pull_capex(codes)
    print(f"capex lake: {len(cap)} reports, {cap['code'].nunique()} names, "
          f"{cap['stat_date'].dt.year.min()} -> {cap['stat_date'].dt.year.max()}")

    done = set(_done_path("capex").read_text(encoding="utf-8").split())
    missing = [c for c in codes if c not in done]
    if missing:
        raise SystemExit(f"ABORT: capex: {len(missing)} codes incomplete "
                         f"(first: {missing[:5]}); re-run to resume")


if __name__ == "__main__":
    main()
