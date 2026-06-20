"""Daily paper-trading driver -- run after A-share market close on a trading day.

  python scripts/paper_live.py                 # refresh data to today, step all tiers, report
  python scripts/paper_live.py --no-refresh     # skip the data pull; recompute on current lake
  python scripts/paper_live.py --as-of 2026-03-31 --no-refresh   # historical replay to a date

It (1) refreshes the lake from BaoStock (membership + 前复权 daily bars; same source as
research), then (2) runs the DEPLOYED strategy forward via live.paper.live_step at each
capital tier, recomputing the idempotent ledger from the seed, and (3) prints + persists a
daily report (equity, today's trades, drawdown, effective diversification) under results/paper/.

Scheduling (Windows Task Scheduler), every trading day ~15:30 CST:
  schtasks /Create /SC DAILY /ST 15:35 /TN hermes-paper ^
    /TR "D:\\Anaconda3\\envs\\hermes\\python.exe F:\\hermes-quant\\scripts\\paper_live.py"
(or drive it from the /schedule skill). It is safe to re-run: each run recomputes from the seed.
"""
import argparse

from hermes.live.feed import refresh
from hermes.live.paper import live_step
from hermes.live.strategy import DEPLOYED

TIERS = [5_000, 10_000, 30_000, 100_000, 500_000]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true", help="skip the BaoStock data pull")
    ap.add_argument("--as-of", default=None, help="recompute through this date (YYYY-MM-DD)")
    ap.add_argument("--tiers", type=int, nargs="*", default=TIERS, help="capital tiers (元)")
    args = ap.parse_args()

    if not args.no_refresh:
        print("refreshing lake from BaoStock (membership + 前复权 daily bars)...")
        refresh(end=args.as_of)

    print(f"\ndeployed = value + 1m-reversal {int(DEPLOYED.value_weight)}/"
          f"{int(DEPLOYED.reversal_weight)}, top{DEPLOYED.n_hold} monthly (PIT HS300, A-share frictions)")
    print(f"  {'tier':>9} {'as_of':>12} {'equity':>14} {'totRet':>8} {'maxDD':>8} "
          f"{'avgN':>6} {'pos':>4} {'todayTrades':>12}")
    last_report = None
    for cap in args.tiers:
        r = live_step(cap, as_of=args.as_of)
        last_report = r
        print(f"  {cap:>9,} {r['as_of']:>12} {r['equity']:>14,.0f} {r['total_return']:>+8.1%} "
              f"{r['max_drawdown']:>8.1%} {r['avg_names_held']:>6.2f} {r['n_positions']:>4} "
              f"{len(r['today_trades']):>12}")

    if last_report and last_report["today_trades"]:
        print(f"\ntoday's fills @ {last_report['as_of']} (largest tier): rebalance executed")
        for t in last_report["today_trades"][:12]:
            side = "BUY " if t["shares"] > 0 else "SELL"
            print(f"  {side} {t['code']} {abs(t['shares']):>6} @ {t['price']:.2f} fee {t['fee']:.2f}")
    else:
        print("\nno fills today (between monthly rebalances) -- holding.")
    print("\nreports + curves + trade logs saved under results/paper/. Re-run safe (recompute-from-seed).")


if __name__ == "__main__":
    main()
