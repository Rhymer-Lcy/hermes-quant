"""Daily paper-trading driver -- run after A-share market close on a trading day.

  python scripts/paper_live.py                 # refresh data to today, step all tiers, report
  python scripts/paper_live.py --no-refresh     # skip the data pull; recompute on current lake
  python scripts/paper_live.py --as-of 2026-03-31 --no-refresh   # historical replay to a date

It (1) refreshes the lake from BaoStock (membership + 前复权 daily bars; same source as
research), then (2) runs the DEPLOYED strategy forward via live.paper.live_step at each
capital tier, recomputing the idempotent ledger from the seed, and (3) prints + persists a
daily report (equity, today's trades, drawdown, effective diversification) under results/paper/.

Scheduling (Windows Task Scheduler), weekdays after close (~15:35 CST). Use the wrapper
scripts/paper_live.ps1 so stdout/stderr are captured to a timestamped log and the exit code
(nonzero on a degraded pull or crash) is visible as the task's last result:
  schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 15:35 /TN hermes-paper ^
    /TR "powershell -NoProfile -ExecutionPolicy Bypass -File F:\\hermes-quant\\scripts\\paper_live.ps1"
It is safe to re-run: each run recomputes from the seed; a holiday/stale run is flagged (fresh=false).
"""
import argparse

from hermes.live.feed import refresh
from hermes.live.paper import live_step
from hermes.live.strategy import ALL_TIERS, DEPLOYED, TIER_LABEL


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true", help="skip the BaoStock data pull")
    ap.add_argument("--as-of", default=None, help="recompute through this date (YYYY-MM-DD)")
    ap.add_argument("--tiers", type=int, nargs="*", default=ALL_TIERS, help="capital tiers (元)")
    args = ap.parse_args()

    if not args.no_refresh:
        print("refreshing lake from BaoStock (membership + 前复权 daily bars)...")
        refresh(end=args.as_of)

    print(f"\ndeployed = value + 1m-reversal {int(DEPLOYED.value_weight)}/"
          f"{int(DEPLOYED.reversal_weight)}, top{DEPLOYED.n_hold} monthly (PIT HS300, A-share frictions)")
    print(f"  {'band':>6} {'tier':>9} {'as_of':>12} {'equity':>14} {'totRet':>8} {'maxDD':>8} "
          f"{'avgN':>6} {'pos':>4} {'todayTrades':>12}")
    last_report = None
    for cap in args.tiers:
        r = live_step(cap, as_of=args.as_of)
        last_report = r
        print(f"  {TIER_LABEL.get(cap, '?'):>6} {cap:>9,} {r['as_of']:>12} {r['equity']:>14,.0f} "
              f"{r['total_return']:>+8.1%} {r['max_drawdown']:>8.1%} {r['avg_names_held']:>6.2f} "
              f"{r['n_positions']:>4} {len(r['today_trades']):>12}")

    if last_report and last_report["today_trades"]:
        print(f"\ntoday's fills @ {last_report['as_of']} (largest tier): rebalance executed")
        for t in last_report["today_trades"][:12]:
            side = "BUY " if t["shares"] > 0 else "SELL"
            print(f"  {side} {t['code']} {abs(t['shares']):>6} @ {t['price']:.2f} fee {t['fee']:.2f}")
    else:
        print("\nno fills today (between monthly rebalances) -- holding.")

    if last_report and not last_report["fresh"]:
        print(f"\n*** STALE: last data bar {last_report['as_of']} is {last_report['lake_lag_days']}d "
              f"behind run date {last_report['run_date']} -- no fresh trading-day data (holiday/"
              f"weekend, source lag, or pull skipped). Record was re-computed, not updated. ***")
    print("\nreports + curves + trade logs saved under results/paper/. Re-run safe (recompute-from-seed).")
    print(f"OK {last_report['run_date'] if last_report else ''} (as_of {last_report['as_of'] if last_report else '-'})")


if __name__ == "__main__":
    import sys
    try:
        main()
    except Exception as exc:                       # unattended run: fail loud + nonzero exit
        print(f"\nERROR: paper_live failed: {exc}", file=sys.stderr)
        sys.exit(1)
