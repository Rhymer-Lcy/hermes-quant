"""Build the CSI 500 survivorship-free dataset for the universe-expansion line.

(1) month-end PIT membership via BaoStock query_zz500_stocks (free + PIT), (2) incrementally
pull forward-adjusted daily bars for the union names NOT already on disk (the HS300 overlap is reused).
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
        n_err = int((~summary["status"].eq("ok")).sum())
        if n_err:
            print(f"  WARNING: {n_err}/{len(new)} names errored; first: "
                  f"{summary.loc[~summary['status'].eq('ok'), 'status'].iloc[0]}")

    # COVERAGE GATE. A mid-batch connection cascade can fail hundreds of pulls while the build
    # still exits 0 (a surviving pull summary records one such run: 28 ok of 886), and a study run
    # against the resulting partial lake does not merely degrade -- at 35% coverage the A6 backtest
    # INVERTED its verdict (CSI500 value Calmar 0.32 against the true 0.10). A dataset build must
    # not report a success it did not achieve.
    on_disk = {f.stem for f in daily.glob("*.parquet")}
    covered = sum(1 for c in union if c.replace(".", "_") in on_disk)
    coverage = covered / len(union)
    print(f"coverage: {covered}/{len(union)} union names with daily bars ({coverage:.1%})")
    if coverage < 0.99:
        raise RuntimeError(
            f"CSI500 dataset INCOMPLETE: only {covered}/{len(union)} names have bars "
            f"({coverage:.1%} < 99%). Re-run this script to pull the missing names; do NOT run "
            f"csi500_universe_study.py / cadence_universe_study.py against a partial dataset.")
    print("CSI500 dataset ready.")


if __name__ == "__main__":
    main()
