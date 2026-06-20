"""Build the CSI500 (中证500) survivorship-free dataset for the universe-expansion line.

(1) month-end PIT membership via BaoStock query_zz500_stocks (free + PIT), (2) incrementally
pull 前复权 daily bars for the union names NOT already on disk (the HS300 overlap is reused).
Free, no token. The long pole is the per-name BaoStock pull -- run it once, in the background.

    python scripts/build_csi500_dataset.py
"""
from hermes.data import ingest
from hermes.data import membership as mb
from hermes.paths import PARQUET_DIR


def main() -> None:
    mdf, union = mb.build_csi500_membership()
    daily = PARQUET_DIR / "daily"
    existing = {f.stem for f in daily.glob("*.parquet")}            # e.g. 'sh_600000'
    new = [c for c in union if c.replace(".", "_") not in existing]
    print(f"CSI500 union {len(union)} names; {len(union) - len(new)} already on disk (HS300 overlap); "
          f"pulling {len(new)} new...")
    if new:
        summary = ingest.pull_universe(new)
        ingest.write_pull_summary(summary, name="csi500")
    print("CSI500 dataset ready.")


if __name__ == "__main__":
    main()
