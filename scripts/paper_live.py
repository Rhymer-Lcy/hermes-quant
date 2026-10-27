"""Daily paper-trading driver -- run after A-share market close on a trading day.

  python scripts/paper_live.py                 # refresh, verify the session, step all tiers, persist
  python scripts/paper_live.py --no-refresh     # no data pull; persist only if the lake holds the session
  python scripts/paper_live.py --as-of 2026-03-31 --no-refresh                  # replay, print only
  python scripts/paper_live.py --as-of 2026-03-31 --no-refresh --out-dir <dir>  # replay into <dir>

It (1) refreshes the lake from BaoStock (membership + forward-adjusted daily bars; same source as
research), (2) verifies that the lake holds the session that is due in Beijing time, from a validated
exchange calendar (live.session; never the machine's local date), then (3) runs the DEPLOYED strategy
forward via live.paper.live_step at each capital tier, recomputing the idempotent ledger from the
seed, and (4) prints + persists a daily report under results/paper/.

Evidence (report field `evidence`):
  current_session      the lake holds the latest session whose publication is due (or an earlier-
                       published later one); the only state in which the canonical record is written
  historical_recompute an explicit --as-of replay or --backtest; never written to the canonical record
  stale_source /       the lake ends before the due session, or too few current members carry its bar:
  incomplete_publication  nothing is written and the run exits 75 so the wrapper retries
A --as-of replay used to overwrite results/paper/ with a ledger cut at that date; it now prints only,
or writes to an explicit --out-dir that is not the canonical directory. It also requires --no-refresh:
a replay never pulls BaoStock, extends the membership or rewrites any lake file.

Right edge: a held name whose series ends is liquidated by the engine at its last price, on the first
bar it lacks (report field `right_edge_exits`, every such exit in the record, not only today's). On
the live path every such name must be a delisting BaoStock itself records (stock_basic status 0,
outDate on or before the session); a failed pull or a bar not yet published blocks the write with
exit 75 instead of recording a sell the market did not print.

Scheduling (Windows Task Scheduler), weekdays in the Beijing evening, after the EOD publication is due
(live.session.PUBLICATION_DUE). Use the wrapper scripts/paper_live.ps1 so stdout/stderr are captured to
a timestamped log and the exit code is visible as the task's last result. It is safe to re-run: each
run recomputes from the seed.

Exit codes: 0 = success; 75 (EX_TEMPFAIL) = TRANSIENT data failure (BaoStock unreachable, the session
not yet published or incomplete -- BaoStockUnavailable), which paper_live.ps1 retries with backoff; any
other nonzero = FATAL (fail loud, no retry; e.g. a trade calendar that contradicts the lake).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from hermes.data.membership import MEMBERSHIP_PARQUET
from hermes.data.sources.baostock_source import BaoStockUnavailable
from hermes.live import session as ses
from hermes.live.feed import assert_right_edge_safe, current_members, refresh, verify_session
from hermes.live.paper import live_step, verify_canonical_manifest, write_canonical_manifest
from hermes.live.strategy import ALL_TIERS, DEPLOYED, PAPER_INCEPTION, TIER_LABEL
from hermes.paths import BACKTESTS_DIR, PAPER_DIR


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true", help="skip the BaoStock data pull")
    ap.add_argument("--as-of", default=None, help="recompute through this date (YYYY-MM-DD)")
    ap.add_argument("--out-dir", default=None,
                    help="write a --as-of replay here (never the canonical results/paper/)")
    ap.add_argument("--tiers", type=int, nargs="*", default=ALL_TIERS, help="capital tiers (CNY)")
    ap.add_argument("--backtest", action="store_true",
                    help="full-history backtest (seed at 2015, inception=None) -> results/backtests/, "
                         "NOT the forward paper record")
    args = ap.parse_args(argv)
    if args.as_of is not None and not args.no_refresh:
        ap.error("--as-of replays the lake as it is and needs --no-refresh: a replay must never pull "
                 "BaoStock, extend the membership or rewrite any lake file")

    inception = None if args.backtest else PAPER_INCEPTION       # default forward paper; --backtest = full history
    live = not args.backtest and args.as_of is None
    if args.backtest:
        out_dir, persist = BACKTESTS_DIR / "deployed_full_history", True
    elif not live:
        out_dir = Path(args.out_dir) if args.out_dir else None
        if out_dir is not None and out_dir.resolve() == Path(PAPER_DIR).resolve():
            ap.error("--out-dir must not be the canonical results/paper/ directory")
        persist = out_dir is not None
    else:
        if args.out_dir:
            ap.error("--out-dir applies only to a --as-of replay")
        out_dir, persist = None, True
    status = {"evidence": ses.HISTORICAL_RECOMPUTE} if not live else None
    summary = None                                    # this run's pull summary (per-code status)

    if not args.no_refresh:
        print("refreshing lake from BaoStock (membership + forward-adjusted daily bars)...")
        result = refresh(end=args.as_of)
        summary = result.summary
        if live:
            status = result.check.report_fields()
    elif live:                                        # recompute on the lake as it is: verify it first
        now = ses.beijing_now()
        cal = ses.fetch_calendar(now)
        status = verify_session(cal, now, current_members(pd.read_parquet(MEMBERSHIP_PARQUET))).report_fields()

    print(f"\ndeployed = value + 1m-reversal {int(DEPLOYED.value_weight)}/"
          f"{int(DEPLOYED.reversal_weight)}, top{DEPLOYED.n_hold} monthly (PIT HS300, A-share frictions)")
    print("mode:", "FULL-HISTORY BACKTEST (seed 2015)" if args.backtest
          else f"PAPER forward record (seed at inception {PAPER_INCEPTION})")
    print(f"evidence: {status['evidence']}" + (
        f" (session {status['session_date']}, {status['session_coverage']:.1%} of current members, "
        f"checked {status['session_checked_at']})" if status.get("session_date") else ""))
    print(f"  {'band':>6} {'tier':>9} {'as_of':>12} {'equity':>14} {'totRet':>8} {'maxDD':>8} "
          f"{'avgN':>6} {'pos':>4} {'todayTrades':>12}")
    # Pass 1 computes every tier without writing. On the live path the canonical record is written
    # (pass 2) only after the right-edge check passes for ALL tiers, so a REFUSAL never leaves some
    # tiers updated and others not; pass 2 must reproduce pass 1 exactly. Pass 2 itself is not a
    # transaction: a crash inside it can leave a torn record, which the manifest written last
    # (canonical_manifest.json, the commit point) exposes until the next successful run rewrites it.
    step = dict(as_of=args.as_of, inception=inception, out_dir=out_dir, status=status)
    reports = {cap: live_step(cap, persist=persist and not live, **step) for cap in args.tiers}
    if live:
        state, problems = verify_canonical_manifest(Path(PAPER_DIR))
        print(f"previous canonical write: {state}" + (f" ({'; '.join(problems[:5])})" if problems else ""))
        exits = sorted({c for r in reports.values() for c in r["right_edge_exits"]})
        if exits:
            disp = assert_right_edge_safe(exits, status["session_date"], summary)
            print(f"right-edge exits verified as delistings: {disp}")
        clock = {"run_date", "lake_lag_days", "fresh"}        # may tick over midnight between passes
        for cap in args.tiers:
            r = live_step(cap, persist=True, **step)
            if {k: v for k, v in r.items() if k not in clock} != \
                    {k: v for k, v in reports[cap].items() if k not in clock}:
                raise RuntimeError(f"tier {cap}: the written report differs from the checked one")
            reports[cap] = r
        write_canonical_manifest(Path(PAPER_DIR), reports)      # the commit point: written last
    last_report = None
    for cap in args.tiers:
        r = reports[cap]
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

    if persist:
        where = "backtests/deployed_full_history" if args.backtest else (args.out_dir or "paper")
        print(f"\nsaved under {'results/' if not args.out_dir else ''}{where}/. Re-run safe (recompute-from-seed).")
    else:
        print("\nhistorical replay: NOT persisted (the canonical record is never rewritten by a replay; "
              "pass --out-dir to write it elsewhere).")
    print(f"OK {last_report['run_date'] if last_report else ''} (as_of {last_report['as_of'] if last_report else '-'})")
    return 0


def run(argv: list[str] | None = None) -> int:
    """main() with the wrapper's exit-code contract: 75 transient, 1 fatal."""
    try:
        return main(argv)
    except BaoStockUnavailable as exc:             # transient (source unreachable / not yet published):
        print(f"\nRETRYABLE: paper_live transient data failure: {exc}", file=sys.stderr)
        return 75                                  # EX_TEMPFAIL -> the wrapper retries with backoff
    except Exception as exc:                       # fatal: fail loud + nonzero exit, no retry
        print(f"\nERROR: paper_live failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run())
