"""Incremental / repair path: re-pull the HS300 union prices WITHOUT rebuilding the
PIT membership. For a fresh build use build_pit_dataset.py (which does membership +
prices in one shot). Reads the union list written by build_membership.

    python scripts/ingest_union.py
"""
import pandas as pd

from hermes.data.ingest import pull_universe, write_pull_summary
from hermes.paths import RAW_DIR


def main() -> None:
    union = pd.read_csv(RAW_DIR / "hs300_union.csv")["code"].astype(str).tolist()
    print(f"re-pulling {len(union)} union names (incl. removed/delisted)...")
    write_pull_summary(pull_universe(union), "union")


if __name__ == "__main__":
    main()
