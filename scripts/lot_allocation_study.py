"""Integer-lot allocation: does closest-weight beat independent flooring? Issue #25.

Pre-registered at https://github.com/Rhymer-Lcy/hermes-quant/issues/25 before any comparative
number existed. The battery frozen there runs here unchanged.

Two comparisons, deliberately separated:

  CONTROLLED  at every historical rebalance, both allocators are scored from the IDENTICAL
              pre-trade book (the baseline path's state). This isolates the allocation decision
              from path divergence and is what the PRIMARY verdict rests on.
  REALISED    each allocator also runs end to end on its own path, giving the turnover, cost and
              return a real account would have seen. SECONDARY evidence only.

Window is strictly before the 2026-06-18 paper inception. Nothing else about the strategy moves.

    conda activate hermes
    python scripts/lot_allocation_study.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.live.strategy import ALL_TIERS, DEPLOYED, PAPER_INCEPTION, deployed_signal
from hermes.paths import RESULTS_DIR, ensure_dirs
from hermes.research.backtest.frictions import AShareCosts
from hermes.research.backtest.lots import (baseline_lots, build_inputs, exhaustive,
                                           ideal_lots, solve, solve_lambda_scan)
from hermes.research.backtest.portfolio import signal_portfolio_backtest

INCEPTION = pd.Timestamp(PAPER_INCEPTION)
START = pd.Timestamp("2015-01-05")
OUT = RESULTS_DIR / "reviews"
COSTS = AShareCosts()
N_HOLD = DEPLOYED.n_hold

# Frozen acceptance bounds from issue #25.
BOUND_COST_BP = 3.0          # median incremental cost increase, bp of equity per rebalance
BOUND_TURNOVER_REL = 0.20    # median relative turnover increase
BOUND_MAXDEV_PP = 2.0        # median max single-name deviation increase, percentage points
BOUND_OVER2X_SHARE = 0.05    # share of rebalances with any name above 2x its target weight


def score(inp, shares: np.ndarray) -> dict:
    """All pre-registered allocation-quality metrics for one candidate book."""
    from hermes.research.backtest.lots import _incremental_cost, _trade_cash, _turnover
    a = shares * inp.price / inp.equity
    d = a - inp.target_w
    tgt = inp.target_w > 0
    a_cash = 1.0 - float(a.sum())
    f = float((d * d).sum() + (a_cash - inp.target_cash_w) ** 2)
    return {
        "te": float(np.sqrt(f)), "f": f,
        "cash_ratio": a_cash,
        "n_held": int((shares[tgt] > 0).sum()),
        "n_target": int(tgt.sum()),
        "max_dev": float(np.abs(d[tgt]).max()) if tgt.any() else 0.0,
        "agg_dev": float(np.abs(d[tgt]).sum()) if tgt.any() else 0.0,
        "inc_cost": _incremental_cost(inp, shares),
        "turnover": _turnover(inp, shares),
        "feasible": bool(_trade_cash(inp, shares) >= -1e-9),
    }


def controlled(log: list[dict], verify_budget: int) -> tuple[pd.DataFrame, dict]:
    """Score baseline and CWIL from the SAME pre-trade state at every rebalance."""
    rows, ver = [], {"tried": 0, "match": 0, "boundary": 0, "skipped": 0}
    for e in log:
        price = pd.Series(e["price"])
        codes = [c for c in e["weights"] if c in price.index]
        extra = sorted(c for c, s in e["positions"].items()
                       if s > 0 and c not in codes and c in price.index)
        codes = codes + extra
        if not codes or e["equity"] <= 0:
            continue
        tw = {c: e["gross"] * e["weights"].get(c, 0.0) * e["scale"] / e["equity"] for c in codes}
        inp = build_inputs(codes, price, tw, e["positions"], e["equity"], e["cash"],
                           COSTS, COSTS.slip, COSTS.lot_size)
        b = baseline_lots(inp)
        c = solve(inp)
        sb, sc = score(inp, b), score(inp, c)
        identical = bool(np.array_equal(b, c))
        if ver["tried"] < verify_budget:
            # Primary verification: the independent parametric-frontier solver.
            alt = solve_lambda_scan(inp)
            if alt is not None:
                ver["tried"] += 1
                fa = score(inp, alt)["f"]
                if abs(fa - sc["f"]) <= 1e-12:
                    ver["match"] += 1
                elif fa < sc["f"]:
                    ver["alt_better"] = ver.get("alt_better", 0) + 1
                else:
                    ver["solve_better"] = ver.get("solve_better", 0) + 1
            # Secondary: true brute force wherever the box is small enough to be tractable.
            bf, at_edge = exhaustive(inp, radius_down=1, radius_up=1)
            if bf is None:
                ver["skipped"] += 1
            elif at_edge:
                ver["boundary"] += 1
            elif score(inp, bf)["f"] >= sc["f"] - 1e-12:
                ver["bf_match"] = ver.get("bf_match", 0) + 1
            else:
                ver["bf_better"] = ver.get("bf_better", 0) + 1
        kx = c / inp.lot
        xi = ideal_lots(inp)
        outside = bool(np.any(kx > np.ceil(xi) + 1e-9) or np.any(kx < np.floor(xi) - 1e-9))
        rows.append({"date": e["date"], "identical": identical, "beyond_floor_ceil": outside,
                     **{f"b_{k}": v for k, v in sb.items()},
                     **{f"c_{k}": v for k, v in sc.items()},
                     "median_px": float(np.median(inp.price[inp.target_w > 0])),
                     "ideal_lots_min": float(np.min(ideal_lots(inp)[inp.target_w > 0]))})
    return pd.DataFrame(rows), ver


def main() -> None:
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof = membership_lookup(mdf)
    close = load_close_panel(codes=union, field="close")
    pe = load_close_panel(codes=union, field="peTTM")
    sig_full = deployed_signal(close, pe, asof)
    hist = close.loc[(close.index >= START) & (close.index < INCEPTION)]
    sig = sig_full.loc[sig_full.index.isin(hist.index)]
    print(f"window {hist.index[0].date()} -> {hist.index[-1].date()} ({len(hist)} bars), "
          f"strictly before the {INCEPTION.date()} inception\n")

    def run(cap, alloc, log=None):
        return signal_portfolio_backtest(hist, sig, cap, N_HOLD, members_asof=asof,
                                         weight_asof=DEPLOYED.weight_asof,
                                         rebalance_band=DEPLOYED.rebalance_band,
                                         collect_trades=True, allocator=alloc, alloc_log=log)

    ctrl, realised, verify = {}, {}, {}
    for cap in ALL_TIERS:
        blog: list = []
        rb = run(cap, None, blog)
        rc = run(cap, "cwil")
        df, ver = controlled(blog, verify_budget=40)
        ctrl[cap], verify[cap] = df, ver
        realised[cap] = {"base": rb, "cwil": rc}
        print(f"  tier {cap:>9,}: {len(df)} rebalances scored, "
              f"exhaustive-verified {ver['match']}/{ver['tried']} "
              f"(box-boundary {ver['boundary']}, untractable {ver['skipped']})")

    # ---------------- exactness verification ----------------
    print(f"\n{'=' * 112}\nSEARCH-SPACE VERIFICATION -- is the heuristic actually finding the "
          f"optimum?\n{'=' * 112}")
    tot_t = sum(v["tried"] for v in verify.values())
    tot_m = sum(v["match"] for v in verify.values())
    tot_b = sum(v["boundary"] for v in verify.values())
    alt_b = sum(v.get("alt_better", 0) for v in verify.values())
    sol_b = sum(v.get("solve_better", 0) for v in verify.values())
    bf_m = sum(v.get("bf_match", 0) for v in verify.values())
    bf_b = sum(v.get("bf_better", 0) for v in verify.values())
    print(f"  Independent parametric-frontier solver run on {tot_t} rebalances: identical objective "
          f"in {tot_m} ({tot_m / max(tot_t, 1):.1%});")
    print(f"    frontier solver strictly better in {alt_b}; local search strictly better in {sol_b}.")
    print(f"  True brute force (floor+/-1 box, tractable cases only): {bf_m} agree, {bf_b} found "
          f"better, {tot_b} inconclusive (optimum on the box edge).")

    # ---------------- is floor/ceil sufficient? ----------------
    print(f"\n{'=' * 112}\nIS floor/ceil A SUFFICIENT SEARCH SPACE? (pre-registration asked for "
          f"proof, not assumption)\n{'=' * 112}")
    beyond = {}
    for cap in ALL_TIERS:
        d = ctrl[cap]
        nb, nt = int(d["beyond_floor_ceil"].sum()), int(len(d))
        beyond[cap] = (nb, nt)
        print(f"  {cap:>9,}: optimum left the floor/ceil box on {nb}/{nt} rebalances "
              f"({nb / max(nt, 1):.1%})")
    print("  A non-zero count is a PROOF BY COUNTER-EXAMPLE that restricting each name to "
          "floor/ceil around its ideal\n  lot count would have missed the optimum.")

    # ---------------- primary: allocation quality ----------------
    print(f"\n{'=' * 112}\nPRIMARY -- ALLOCATION QUALITY, controlled (identical pre-trade book)"
          f"\n{'=' * 112}")
    print(f"  {'tier':>9} | {'tracking error':>21} | {'post-reb cash':>21} | "
          f"{'names held (med)':>17} | {'max dev (med)':>15} | {'identical':>9}")
    print(f"  {'':>9} | {'base med':>10} {'cwil med':>10} | {'base med':>10} {'cwil med':>10} | "
          f"{'base':>8} {'cwil':>8} | {'base':>7} {'cwil':>7} | {'':>9}")
    summary = []
    for cap in ALL_TIERS:
        d = ctrl[cap]
        row = {
            "cap": cap, "n": int(len(d)),
            "te_b": float(d["b_te"].median()), "te_c": float(d["c_te"].median()),
            "te_b90": float(d["b_te"].quantile(0.9)), "te_c90": float(d["c_te"].quantile(0.9)),
            "cash_b": float(d["b_cash_ratio"].median()), "cash_c": float(d["c_cash_ratio"].median()),
            "cash_b90": float(d["b_cash_ratio"].quantile(0.9)),
            "cash_c90": float(d["c_cash_ratio"].quantile(0.9)),
            "names_b": float(d["b_n_held"].median()), "names_c": float(d["c_n_held"].median()),
            "all10_b": float((d["b_n_held"] >= d["b_n_target"]).mean()),
            "all10_c": float((d["c_n_held"] >= d["c_n_target"]).mean()),
            "maxdev_b": float(d["b_max_dev"].median()), "maxdev_c": float(d["c_max_dev"].median()),
            "aggdev_b": float(d["b_agg_dev"].median()), "aggdev_c": float(d["c_agg_dev"].median()),
            "cost_b": float(d["b_inc_cost"].median()), "cost_c": float(d["c_inc_cost"].median()),
            "turn_b": float(d["b_turnover"].median()), "turn_c": float(d["c_turnover"].median()),
            "identical": float(d["identical"].mean()),
            "over2x_c": float(d["c_max_dev"].gt(0).mean()),
        }
        summary.append(row)
        print(f"  {cap:>9,} | {row['te_b']:>10.4f} {row['te_c']:>10.4f} | "
              f"{row['cash_b']:>10.2%} {row['cash_c']:>10.2%} | "
              f"{row['names_b']:>8.1f} {row['names_c']:>8.1f} | "
              f"{row['maxdev_b']:>7.2%} {row['maxdev_c']:>7.2%} | {row['identical']:>9.1%}")

    print("\n  p90 tracking error and p90 cash:")
    print(f"  {'tier':>9} {'TE p90 base':>12} {'TE p90 cwil':>12} {'cash p90 base':>14} "
          f"{'cash p90 cwil':>14} {'all-10 base':>12} {'all-10 cwil':>12}")
    for r in summary:
        print(f"  {r['cap']:>9,} {r['te_b90']:>12.4f} {r['te_c90']:>12.4f} "
              f"{r['cash_b90']:>14.2%} {r['cash_c90']:>14.2%} {r['all10_b']:>12.1%} "
              f"{r['all10_c']:>12.1%}")

    # ---------------- frozen acceptance bounds ----------------
    print(f"\n{'=' * 112}\nFROZEN ACCEPTANCE BOUNDS (issue #25)\n{'=' * 112}")
    checks = []
    for r in summary:
        cap = r["cap"]
        d = ctrl[cap]
        cost_bp = (r["cost_c"] - r["cost_b"]) / cap * 1e4
        turn_rel = (r["turn_c"] - r["turn_b"]) / max(r["turn_b"], 1e-9)
        maxdev_pp = (r["maxdev_c"] - r["maxdev_b"]) * 100
        tgt_w = 1.0 / N_HOLD
        over2x = float((d["c_max_dev"] > tgt_w).mean())
        c = {
            "cap": cap,
            "te_med_better": r["te_c"] <= r["te_b"],
            "te_p90_better": r["te_c90"] <= r["te_b90"],
            "cash_lower": r["cash_c"] <= r["cash_b"],
            "names_not_lower": r["names_c"] >= r["names_b"],
            "cost_ok": cost_bp <= BOUND_COST_BP, "cost_bp": cost_bp,
            "turn_ok": turn_rel <= BOUND_TURNOVER_REL, "turn_rel": turn_rel,
            "maxdev_ok": maxdev_pp <= BOUND_MAXDEV_PP, "maxdev_pp": maxdev_pp,
            "over2x_ok": over2x <= BOUND_OVER2X_SHARE, "over2x": over2x,
        }
        c["all_pass"] = all(c[k] for k in ("te_med_better", "te_p90_better", "cash_lower",
                                           "names_not_lower", "cost_ok", "turn_ok",
                                           "maxdev_ok", "over2x_ok"))
        checks.append(c)
        print(f"  {cap:>9,}: TEmed {'ok' if c['te_med_better'] else 'FAIL':>4} "
              f"TEp90 {'ok' if c['te_p90_better'] else 'FAIL':>4} "
              f"cash {'ok' if c['cash_lower'] else 'FAIL':>4} "
              f"names {'ok' if c['names_not_lower'] else 'FAIL':>4} | "
              f"cost {cost_bp:>+6.2f}bp {'ok' if c['cost_ok'] else 'FAIL':>4} | "
              f"turn {turn_rel:>+7.1%} {'ok' if c['turn_ok'] else 'FAIL':>4} | "
              f"maxdev {maxdev_pp:>+5.2f}pp {'ok' if c['maxdev_ok'] else 'FAIL':>4} | "
              f"over2x {over2x:>5.1%} {'ok' if c['over2x_ok'] else 'FAIL':>4} "
              f"-> {'PASS' if c['all_pass'] else 'FAIL'}")
    verdict_all = all(c["all_pass"] for c in checks)
    print(f"\n  ALL SEVEN TIERS PASS: {verdict_all}")

    # ---------------- secondary: realised path ----------------
    print(f"\n{'=' * 112}\nSECONDARY (diagnostic only, cannot decide the verdict) -- realised path"
          f"\n{'=' * 112}")
    print(f"  {'tier':>9} {'CAGR base':>10} {'CAGR cwil':>10} {'maxDD base':>11} "
          f"{'maxDD cwil':>11} {'costs base':>12} {'costs cwil':>12} {'trades b':>9} "
          f"{'trades c':>9} {'names b':>8} {'names c':>8}")
    sec = []
    for cap in ALL_TIERS:
        b, c = realised[cap]["base"], realised[cap]["cwil"]
        sec.append({"cap": cap, "cagr_b": b.cagr, "cagr_c": c.cagr,
                    "dd_b": b.max_drawdown, "dd_c": c.max_drawdown,
                    "cost_b": b.total_costs, "cost_c": c.total_costs,
                    "tr_b": len(b.trades), "tr_c": len(c.trades),
                    "nm_b": b.avg_names_held, "nm_c": c.avg_names_held})
        print(f"  {cap:>9,} {b.cagr:>10.3%} {c.cagr:>10.3%} {b.max_drawdown:>11.2%} "
              f"{c.max_drawdown:>11.2%} {b.total_costs:>12,.0f} {c.total_costs:>12,.0f} "
              f"{len(b.trades):>9} {len(c.trades):>9} {b.avg_names_held:>8.2f} "
              f"{c.avg_names_held:>8.2f}")

    # ---------------- falsification ----------------
    print(f"\n{'=' * 112}\nFALSIFICATION -- where does the candidate HURT?\n{'=' * 112}")
    neg = []
    for cap in ALL_TIERS:
        d = ctrl[cap]
        worse_te = d[d["c_te"] > d["b_te"] + 1e-12]
        worse_cash = d[d["c_cash_ratio"] > d["b_cash_ratio"] + 1e-12]
        near0 = d[d["b_cash_ratio"] < 0.005]
        near0_worse = near0[near0["c_te"] > near0["b_te"] + 1e-12]
        hi_px = d[d["median_px"] > d["median_px"].quantile(0.8)]
        lo_px = d[d["median_px"] < d["median_px"].quantile(0.2)]
        neg.append({"cap": cap, "worse_te": float(len(worse_te) / len(d)),
                    "worse_cash": float(len(worse_cash) / len(d)),
                    "near0_n": int(len(near0)),
                    "near0_worse": float(len(near0_worse) / max(len(near0), 1)),
                    "te_gain_hi_px": float((hi_px["b_te"] - hi_px["c_te"]).median()),
                    "te_gain_lo_px": float((lo_px["b_te"] - lo_px["c_te"]).median())})
        print(f"  {cap:>9,}: TE worse on {len(worse_te) / len(d):>5.1%} of rebalances; "
              f"cash worse on {len(worse_cash) / len(d):>5.1%}; "
              f"of {len(near0):>3} already-near-zero-cash rebalances "
              f"{len(near0_worse) / max(len(near0), 1):>5.1%} got worse TE; "
              f"median TE gain high-price baskets {(hi_px['b_te'] - hi_px['c_te']).median():+.4f} "
              f"vs low-price {(lo_px['b_te'] - lo_px['c_te']).median():+.4f}")

    # concentration: does CWIL tilt small accounts into cheap names?
    print("\n  concentration check -- largest single-name weight deviation vs the 10% target:")
    for cap in ALL_TIERS:
        d = ctrl[cap]
        print(f"  {cap:>9,}: baseline median max weight deviation {d['b_max_dev'].median():.2%}, "
              f"candidate {d['c_max_dev'].median():.2%}; candidate p99 {d['c_max_dev'].quantile(0.99):.2%}")

    for cap in ALL_TIERS:
        ctrl[cap].assign(cap=cap).to_parquet(OUT / f"lot_ctrl_{cap}.parquet", index=False)
    (OUT / "lot_allocation_study.json").write_text(json.dumps(
        {"window": [str(hist.index[0].date()), str(hist.index[-1].date())],
         "summary": summary, "checks": checks, "secondary": sec, "negative": neg,
         "verify": {str(k): v for k, v in verify.items()},
         "beyond_floor_ceil": {str(k): list(v) for k, v in beyond.items()},
         "verdict_all_tiers_pass": bool(verdict_all)}, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {OUT / 'lot_allocation_study.json'}")


if __name__ == "__main__":
    main()
