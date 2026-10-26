"""Repair path: re-pull the HS300 union's forward-adjusted daily bars in full, WITHOUT rebuilding the
PIT membership. For a fresh build use build_pit_dataset.py (membership + prices in one shot). Reads
the union list written by build_membership.

    python scripts/ingest_union.py --end 2026-10-26        # --end is required; there is no default

There is deliberately no implicit end date: the library default (BACKTEST_END, 2025-12-31) once made
this script a way to cut every current series back to 2025. Every write also goes through
ingest.write_full_series, which refuses to replace a file with a series that lacks any bar the lake
already holds, so an end date earlier than the stored data, a later --start, or a vendor gap leaves
that file untouched and is reported as an error. There is no override. Exit status 0 only when
every code was pulled and written; 1 when any code failed or was refused; 2 on bad arguments.
"""
import argparse
import sys
from datetime import datetime

import pandas as pd

from hermes.data.ingest import BACKTEST_START, pull_universe, write_pull_summary
from hermes.paths import RAW_DIR


def _iso_date(text: str) -> str:
    try:
        return datetime.strptime(text, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a YYYY-MM-DD date: {text!r}") from exc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--end", required=True, type=_iso_date,
                    help="last date to pull, YYYY-MM-DD (required; normally today's date)")
    ap.add_argument("--start", default=BACKTEST_START, type=_iso_date,
                    help=f"first date to pull, YYYY-MM-DD (default {BACKTEST_START})")
    args = ap.parse_args(argv)
    if args.end < args.start:
        ap.error(f"--end {args.end} is before --start {args.start}")

    union = pd.read_csv(RAW_DIR / "hs300_union.csv")["code"].astype(str).tolist()
    print(f"re-pulling {len(union)} union names (incl. removed/delisted), {args.start} .. {args.end}...")
    summary = pull_universe(union, args.start, args.end)
    write_pull_summary(summary, "union")
    failed = summary[(summary["status"] != "ok") | (summary["rows"] == 0)]   # an empty pull wrote nothing
    if len(failed):
        print(f"{len(failed)} of {len(summary)} codes were not written (see the errors above)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
