"""Pull daily history for the full HS300 union (all names ever in the index),
including removed/delisted names, into the parquet lake.

Run AFTER build_membership has written data/raw/hs300_union.csv.

    python scripts/ingest_union.py
"""
import pandas as pd

from hermes.data.ingest import pull_universe
from hermes.paths import RAW_DIR


def main() -> None:
    union = pd.read_csv(RAW_DIR / "hs300_union.csv")["code"].astype(str).tolist()
    print(f"pulling {len(union)} union names (incl. removed/delisted)...")
    summary = pull_universe(union)
    ok = int((summary["status"] == "ok").sum())
    print(f"pulled {ok}/{len(summary)} symbols")
    summary.to_csv(RAW_DIR / "union_pull_summary.csv", index=False)
    errors = summary[summary["status"] != "ok"]
    if not errors.empty:
        print("\nsymbols with errors (likely never-traded codes):")
        print(errors.to_string(index=False))


if __name__ == "__main__":
    main()
