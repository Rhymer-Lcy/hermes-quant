"""Pull Tushare daily_basic (total_mv etc.) for the HS300 union -> parquet lake,
to add the size factor. Run after build_pit_dataset (needs data/raw/hs300_union.csv).

    python scripts/ingest_size.py

WARNING (current Tushare tier): daily_basic is rate-limited to ~1 call/minute (and a
~1 call/hour cap was also observed), so a 657-name bulk pull is impractical here -- the
size factor is DEFERRED until the token tier is upgraded or a free daily market-cap
source is wired in. The adapter is kept ready; the rest of the factor stack uses
BaoStock (value/momentum/low-vol/reversal), which has no such limit.
"""
import pandas as pd

from hermes.data.tushare_source import pull_daily_basic
from hermes.paths import RAW_DIR


def main() -> None:
    union = pd.read_csv(RAW_DIR / "hs300_union.csv")["code"].astype(str).tolist()
    print(f"pulling daily_basic for {len(union)} names...")
    summary = pull_daily_basic(union)
    ok = int((summary["status"] == "ok").sum())
    print(f"pulled {ok}/{len(summary)}")
    summary.to_csv(RAW_DIR / "size_pull_summary.csv", index=False)


if __name__ == "__main__":
    main()
