"""D=5 forward-shadow driver -- run AFTER the canonical D=1 paper step succeeds. Issue #22.

  python scripts/paper_shadow_d5.py                  # recompute on the current lake, persist
  python scripts/paper_shadow_d5.py --dry-run        # gates + report, write nothing
  python scripts/paper_shadow_d5.py --as-of 2026-09-30   # replay through a date

Never refreshes the lake: the canonical run owns the data pull, and the shadow reads whatever the
canonical run already validated. That ordering is what keeps a shadow failure from touching the
canonical record -- this script writes only under results/paper_shadow/d5/ and asserts, before and
after, that results/paper/ is byte-for-byte unchanged.

Three blocking gates run before any output is written; any failure exits nonzero having written
nothing:
  1  production parity   -- live_step(persist=False) reproduces the on-disk canonical reports
  2  common-state parity -- the shadow path reproduces the canonical ledger EXACTLY through the
                            2026-08-28 fork bar (equity, cash, positions, shares, trades, costs)
  3  no look-ahead       -- every post-fork rebalance has signal bar < execution bar

Exit codes: 0 success; 75 (EX_TEMPFAIL) transient data failure; any other nonzero = fatal.
The wrapper treats ALL shadow exit codes as non-fatal for the canonical task.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

import pandas as pd

from hermes.live.paper import live_step
from hermes.live.shadow import (BASELINE_DAY, CANDIDATE_DAY, SCHEMA_VERSION, SHADOW_DIR,
                                SHADOW_INCEPTION_ASOF, ShadowGateError,
                                assert_canonical_untouched, build_manifest, build_panels,
                                fingerprint_canonical, run_book, runtime_provenance,
                                shadow_schedule, validate_manifest)
from hermes.live.strategy import ALL_TIERS, TIER_LABEL
from hermes.paths import PAPER_DIR
from hermes.research.backtest.schedule import calendar_rebalance_schedule

TOL = 1e-9


def head_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:                       # noqa: BLE001 -- provenance is best-effort, not a gate
        return "unknown"


def gate_production_parity(tiers) -> dict[int, dict]:
    """GATE 1: the canonical ledger on disk is reproducible by the production path."""
    print("=== GATE 1 (blocking): canonical production parity ===")
    disk = {}
    for cap in tiers:
        ref = json.loads((PAPER_DIR / f"report_{cap}.json").read_text(encoding="utf-8"))
        rep = live_step(cap, persist=False)
        ok = (rep["as_of"] == ref["as_of"]
              and abs(rep["equity"] - ref["equity"]) < TOL
              and abs(rep["total_return"] - ref["total_return"]) < 1e-12
              and abs(rep["max_drawdown"] - ref["max_drawdown"]) < 1e-12
              and rep["positions"] == ref["positions"]
              and rep["n_trades_total"] == ref["n_trades_total"])
        print(f"  {cap:>9,}: equity {rep['equity']:>14,.2f}  "
              f"d {rep['equity'] - ref['equity']:+.2e}  -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise ShadowGateError(f"GATE 1 failed at tier {cap}")
        disk[cap] = ref
    return disk


def gate_common_state(panels, disk, tiers) -> None:
    """GATE 2: through the fork bar the shadow book IS the canonical book, tier by tier."""
    close, signal, asof_fn = panels
    fork = pd.Timestamp(SHADOW_INCEPTION_ASOF)
    pre = close.index <= fork
    sch_shadow = shadow_schedule(close.index)
    sch_base = calendar_rebalance_schedule(close.index, BASELINE_DAY)
    pre_shadow = {e: s for e, s in sch_shadow.items() if close.index[e] <= fork}
    pre_base = {e: s for e, s in sch_base.items() if close.index[e] <= fork}
    print(f"\n=== GATE 2 (blocking): common state at {SHADOW_INCEPTION_ASOF} ===")
    if pre_shadow != pre_base:
        raise ShadowGateError("shadow and canonical schedules differ BEFORE the fork bar")
    print("  pre-fork schedules identical: PASS")
    for cap in tiers:
        a = run_book(close.loc[pre], signal.loc[pre], cap, asof_fn, pre_shadow)
        b = run_book(close.loc[pre], signal.loc[pre], cap, asof_fn, pre_base)
        pos_a, pos_b = _fold(a.trades), _fold(b.trades)
        cash_a = float(a.equity.iloc[-1]) - _mark(pos_a, close.loc[pre].iloc[-1])
        cash_b = float(b.equity.iloc[-1]) - _mark(pos_b, close.loc[pre].iloc[-1])
        ok = (abs(float(a.equity.iloc[-1]) - float(b.equity.iloc[-1])) < TOL
              and pos_a == pos_b and abs(cash_a - cash_b) < 1e-6
              and len(a.trades) == len(b.trades)
              and abs(a.total_costs - b.total_costs) < TOL
              and abs(float(a.equity.iloc[-1]) - disk[cap]["equity"]) < TOL
              and pos_a == {k: int(v) for k, v in disk[cap]["positions"].items()})
        print(f"  {cap:>9,}: equity {float(a.equity.iloc[-1]):>14,.2f}  cash {cash_a:>12,.2f}  "
              f"names {len(pos_a):>2}  trades {len(a.trades):>3}  costs {a.total_costs:>9,.2f}"
              f"  -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise ShadowGateError(f"GATE 2 failed at tier {cap}")


def _fold(trades) -> dict[str, int]:
    pos: dict[str, int] = {}
    for t in trades:
        pos[t["code"]] = pos.get(t["code"], 0) + int(t["shares"])
    return {c: s for c, s in sorted(pos.items()) if s > 0}


def _mark(pos, row) -> float:
    return float(sum(sh * row.get(c, 0.0) for c, sh in pos.items()))


def gate_no_lookahead(panels) -> None:
    """GATE 3: every post-fork rebalance reads a strictly earlier bar than it trades on."""
    close, _s, _a = panels
    fork = pd.Timestamp(SHADOW_INCEPTION_ASOF)
    sch = shadow_schedule(close.index)                 # itself calls assert_no_lookahead
    future = {e: s for e, s in sch.items() if close.index[e] > fork}
    print(f"\n=== GATE 3 (blocking): no look-ahead on {len(future)} post-fork rebalance(s) ===")
    for e, s in sorted(future.items()):
        if not s < e:
            raise ShadowGateError(f"look-ahead at exec bar {e}")
        print(f"  signal {close.index[s].date()} < execution {close.index[e].date()}  -> PASS")
    if not future:
        print("  none yet (the fork bar is the latest bar) -- vacuously PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="run the gates, write nothing")
    ap.add_argument("--as-of", default=None, help="recompute through this date (YYYY-MM-DD)")
    ap.add_argument("--tiers", type=int, nargs="*", default=ALL_TIERS)
    args = ap.parse_args()

    before = fingerprint_canonical()
    print(f"canonical results/paper fingerprinted: {len(before)} files (write-protected)\n")

    disk = gate_production_parity(args.tiers)
    panels = build_panels(args.as_of)
    gate_common_state(panels, disk, args.tiers)
    gate_no_lookahead(panels)
    print("\n  all three gates PASSED -- shadow may be written")

    # --- runtime provenance: a PERSISTED forward run must come from clean committed code ---
    prov = runtime_provenance()
    print(f"\n=== runtime provenance ===\n  code {prov['runtime_code_sha'][:8]}   "
          f"worktree {'clean' if prov['runtime_worktree_clean'] else 'DIRTY'}")
    if not prov["runtime_worktree_clean"] and not args.dry_run:
        raise ShadowGateError(
            "refusing to persist forward evidence from a DIRTY worktree -- the result could not be "
            "traced to any revision. Commit (or stash) first, or use --dry-run. The canonical D=1 "
            "record is unaffected either way.")

    manifest_path = SHADOW_DIR / "manifest.json"
    expected = build_manifest(head_commit())
    on_disk = validate_manifest(manifest_path, expected)
    if not args.dry_run:
        SHADOW_DIR.mkdir(parents=True, exist_ok=True)
        if not manifest_path.exists():                  # frozen on first write, never rewritten
            manifest_path.write_text(json.dumps(expected, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
            print(f"  manifest FROZEN -> {manifest_path}")
        elif on_disk.get("schema_version", 1) < SCHEMA_VERSION:
            # ONE-TIME provenance migration. Permitted only while no post-inception bar exists, and
            # it carries the v1 value forward verbatim rather than discarding it. The experiment
            # definition is untouched -- validate_manifest already refused any frozen-field drift
            # above, so reaching here means candidate, baseline, inception, spec, costs and tiers
            # all still agree.
            migrated = build_manifest(
                head_commit(),
                experiment_freeze_sha=head_commit(),
                legacy_created_at_commit=on_disk.get("created_at_commit"))
            (SHADOW_DIR / "manifest_v1_original.json").write_text(
                json.dumps(on_disk, indent=2, ensure_ascii=False), encoding="utf-8")
            manifest_path.write_text(json.dumps(migrated, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
            print(f"  manifest MIGRATED v{on_disk.get('schema_version', 1)} -> v{SCHEMA_VERSION}; "
                  f"v1 preserved as manifest_v1_original.json")
        else:
            print(f"  manifest validated (frozen at {on_disk.get('experiment_freeze_sha', '?')[:8]}, "
                  f"schema v{on_disk.get('schema_version')})")

    from hermes.live.shadow import shadow_step
    print(f"\nD=5 forward shadow (candidate {CANDIDATE_DAY} vs baseline {BASELINE_DAY}), "
          f"inception {SHADOW_INCEPTION_ASOF}")
    print(f"  {'band':>6} {'tier':>9} {'as_of':>12} {'D5 equity':>14} {'D5 ret':>9} "
          f"{'D1 ret':>9} {'excess':>9} {'D5 maxDD':>9} {'trades':>7}")
    last = None
    for cap in args.tiers:
        r = shadow_step(cap, as_of=args.as_of, persist=not args.dry_run, panels=panels)
        last = r
        print(f"  {TIER_LABEL.get(cap, '?'):>6} {cap:>9,} {r['as_of']:>12} "
              f"{r['equity']:>14,.2f} {r['total_return_since_shadow_inception']:>+9.3%} "
              f"{r['d1_return_since_shadow_inception']:>+9.3%} {r['excess_return_vs_d1']:>+9.3%} "
              f"{r['max_drawdown_since_shadow_inception']:>9.2%} "
              f"{r['n_trades_since_shadow_inception']:>7}")

    assert_canonical_untouched(before)
    print(f"\n  results/paper verified byte-for-byte unchanged ({len(before)} files)")
    if last and not last["fresh"]:
        print(f"\n*** STALE: last bar {last['as_of']} is {last['lake_lag_days']}d behind "
              f"{last['run_date']} -- recomputed, not updated. ***")
    if args.dry_run:
        print("\n  DRY RUN: nothing written")
    else:
        print(f"\n  saved -> {SHADOW_DIR}")
    print("\n  NOTE: this is a FORWARD SHADOW of a historical champion that FAILED issue #21's")
    print("  statistical battery. It is not a recommendation, and it can never rewrite the")
    print(f"  canonical D=1 record under {PAPER_DIR}.")
    print(f"OK {last['run_date'] if last else ''} (as_of {last['as_of'] if last else '-'})")


if __name__ == "__main__":
    from hermes.data.sources.baostock_source import BaoStockUnavailable

    try:
        main()
    except BaoStockUnavailable as exc:
        print(f"\nRETRYABLE: shadow transient data failure: {exc}", file=sys.stderr)
        sys.exit(75)
    except ShadowGateError as exc:
        print(f"\nGATE FAILURE: {exc}\nNo shadow output written.", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:                            # noqa: BLE001 -- fail loud, never silently
        print(f"\nERROR: paper_shadow_d5 failed: {exc}", file=sys.stderr)
        sys.exit(1)
