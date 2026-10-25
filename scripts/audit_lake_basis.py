"""Read-only audit: is every HS300-union series in the daily lake on ONE forward-adjustment basis?

The incremental refresh re-checks only the last ingest.REBASE_WINDOW_BARS stored bars per code, so
a dividend factor the vendor applies later than that would still leave a series on a mixed basis
(the 2026-10-24 audit found 346 of 643 series in that state under the former one-bar check). This
script pulls each code's forward-adjusted bars fresh from BaoStock over a longer window, compares
every stored close with the same rule the refresh uses (ingest._window_rebased), and reports the
codes that differ together with their lake/fresh ratio range. It writes nothing under data/.

    python scripts/audit_lake_basis.py                      # window from Jan 1 of the lake's last year
    python scripts/audit_lake_basis.py --since 2025-01-01 --out results/lake_basis_audit.json
    python scripts/audit_lake_basis.py --codes sh.601601 sz.000001  # only these codes

Exit status: 0 = every audited series matches, 1 = at least one series is off-basis or failed to
pull. Heal an off-basis code by deleting its parquet: the next refresh treats it as a new name and
re-pulls its whole history through today. (Not scripts/ingest_union.py: its pull_universe call
defaults to end=BACKTEST_END and would cut the series off there.) Then re-run this audit. BaoStock
reads its socket without a timeout, so run it under a time limit; the progress line every 50 codes
shows where it stands.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from hermes.data import ingest
from hermes.data.sources import baostock_source as bss
from hermes.paths import PARQUET_DIR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="first date compared (default: Jan 1 of the last year)")
    ap.add_argument("--out", default=None, help="optional JSON report path")
    ap.add_argument("--codes", nargs="*", default=None, help="audit only these union codes")
    args = ap.parse_args()

    daily = PARQUET_DIR / "daily"
    union = sorted(pd.read_parquet(PARQUET_DIR / "hs300_membership.parquet")["code"].unique())
    if args.codes:
        unknown = sorted(set(args.codes) - set(union))
        if unknown:
            print(f"not in the HS300 union: {unknown}")
            return 1
        union = sorted(args.codes)
    off, failed, skipped, audited = [], [], 0, 0
    with bss.session():
        for i, code in enumerate(union, 1):
            path = daily / f"{code.replace('.', '_')}.parquet"
            if not path.exists():
                skipped += 1
                continue
            stored = pd.read_parquet(path, columns=["date", "close"])
            stored["date"] = pd.to_datetime(stored["date"])
            last = stored["date"].max()
            since = pd.Timestamp(args.since) if args.since else pd.Timestamp(year=last.year, month=1, day=1)
            if (stored["date"] >= since).sum() == 0:
                skipped += 1                      # left the market before the window
                continue
            try:
                fresh = ingest._daily_bars_reloginning(code, since.strftime("%Y-%m-%d"),
                                                       last.strftime("%Y-%m-%d"))
            except Exception as exc:  # noqa: BLE001 -- a failed pull is reported, never hidden
                failed.append({"code": code, "error": str(exc)})
                continue
            fresh = pd.DataFrame({"date": pd.to_datetime(fresh["date"]), "close": fresh["close"]})
            audited += 1
            if ingest._window_rebased(stored, fresh, since):
                both = stored[stored["date"] >= since].merge(fresh, on="date", suffixes=("_lake", "_fresh"))
                ratio = both["close_lake"] / both["close_fresh"]
                is_off = (ratio - 1).abs() > 1e-6
                bad = ratio[is_off] if is_off.any() else ratio      # range over the off-basis bars
                off.append({"code": code, "ratio_min": float(bad.min()), "ratio_max": float(bad.max()),
                            "last_off": str(both.loc[is_off, "date"].max().date()) if is_off.any() else None})
            if i % 50 == 0 or i == len(union):
                print(f"  ...{i}/{len(union)} audited, {len(off)} off-basis", flush=True)
            time.sleep(0.2)

    print(f"audited {audited} of {len(union)} union codes ({skipped} without bars in the window), "
          f"{len(off)} off-basis, {len(failed)} failed pulls")
    for r in off:
        print(f"  OFF {r['code']}  lake/fresh {r['ratio_min']:.6f}..{r['ratio_max']:.6f}  "
              f"last off-basis bar {r['last_off']}")
    for r in failed:
        print(f"  FAILED {r['code']}: {r['error']}")
    if args.out:
        Path(args.out).write_text(json.dumps({"audited": audited, "skipped": skipped, "off": off,
                                              "failed": failed}, indent=1), encoding="utf-8")
    return 0 if not off and not failed else 1


if __name__ == "__main__":
    sys.exit(main())
