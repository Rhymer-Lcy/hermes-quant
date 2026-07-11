"""Daily CB double-low paper driver -- run after the A-share close on trading evenings.

  python scripts/cb_paper_live.py               # refresh recent bonds, recompute, report
  python scripts/cb_paper_live.py --no-refresh  # recompute on the current lake
  python scripts/cb_paper_live.py --as-of 2026-08-31 --no-refresh   # historical replay

The refresh is incremental (bonds with a bar inside the last 45 days, plus new listings;
~10-15 min at the polite pull rate) and the record is recompute-from-inception, so a
missed evening loses nothing -- the next run reconstructs every skipped bar. Scheduling:
scripts/schedule_tasks.ps1 registers hermes-cb-paper at 19:40 Beijing on weekdays, via the
wrapper scripts/cb_paper_live.ps1 (log capture + retry).

Exit codes: 0 = success; 75 (EX_TEMPFAIL) = transient -- the universe listing was
unreachable, or the refresh was degraded (>10% of the refresh set missed), so signals
could be stale; the wrapper retries with backoff. Any other nonzero = fatal.
"""
import argparse
import sys

from hermes.cb import data as cbdata
from hermes.cb.paper import CB_PAPER_INCEPTION, N_HOLD, paper_step

EX_TEMPFAIL = 75
DEGRADED_FRACTION = 0.10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true", help="skip the incremental pull")
    ap.add_argument("--as-of", default=None, help="recompute through this date (YYYY-MM-DD)")
    args = ap.parse_args()

    if not args.no_refresh:
        print("refreshing the CB lake (universe + recent bonds, incremental)...", flush=True)
        try:
            out = cbdata.refresh_recent()
        except Exception as exc:
            print(f"RETRYABLE: refresh failed outright: {exc}", file=sys.stderr)
            return EX_TEMPFAIL
        n_miss = len(out["bars"]) + len(out["premium"])
        print(f"refreshed {out['attempted']} bonds, {n_miss} misses")
        if out["attempted"] and n_miss > DEGRADED_FRACTION * 2 * out["attempted"]:
            print(f"RETRYABLE: degraded refresh ({n_miss} misses on {out['attempted']} bonds)",
                  file=sys.stderr)
            return EX_TEMPFAIL

    r = paper_step(as_of=args.as_of)
    print(f"\nCB double-low paper record (top{N_HOLD}, inception {CB_PAPER_INCEPTION})")
    print(f"  as_of {r['as_of']}  equity {r['equity']:.4f}  total {r['total_return']:+.2%}  "
          f"maxDD {r['max_drawdown']:.2%}  positions {r['n_positions']}  "
          f"rebalances {r['n_rebalances']}")
    if r["rebalanced_today"]:
        print(f"  REBALANCED today (signal {r['signal_date']}): "
              f"+{len(r['entered_today'])} / -{len(r['exited_today'])}")
    if not r["fresh"]:
        print(f"\n*** STALE: last bar {r['as_of']} is {r['lake_lag_days']}d behind run date "
              f"{r['run_date']} (holiday/weekend or source lag). Recomputed, not updated. ***")
    print(f"\nsaved under results/paper/ (cb_curve.parquet, cb_report.json). Re-run safe "
          f"(recompute-from-inception).\nOK {r['run_date']} (as_of {r['as_of']})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:                       # unattended: fail loud + nonzero exit
        print(f"ERROR: cb_paper_live failed: {exc}", file=sys.stderr)
        sys.exit(1)
