"""Three-month forward review of the canonical Hermes paper record. Read-only.

Canonical source is `results/paper/` and nothing else is treated as the record. This script
NEVER writes there; it only reads, and it asserts that fact at the end by fingerprinting the
directory before and after.

It answers, in order:

  B1  integrity      -- calendar completeness, rebalance dates, duplicate fills, ledger
                        reconciliation, cost reconciliation, idempotency of the production path
  B2  performance    -- all seven capital tiers, cross-tier dispersion and its causes
  B3  benchmark      -- CSI 300 TOTAL RETURN (H00300, the index provider's own series), with the
                        price index shown alongside to size the dividend gap inside this window
  B4  context        -- where the forward period sits in the distribution of the SAME strategy's
                        pre-inception 67-bar windows (strictly data before 2026-06-18)
  B5  attribution    -- who actually moved the book
  B6  d5 shadow      -- reported separately and never mixed into the D=1 evaluation

Nothing here is an optimisation: no parameter is searched, nothing is selected on the forward
window, and no strategy object is modified.

    conda activate hermes
    python scripts/forward_review.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.live.paper import live_step
from hermes.live.strategy import (ALL_TIERS, DEPLOYED, PAPER_INCEPTION, TIER_LABEL,
                                  deployed_signal)
from hermes.paths import PAPER_DIR, RESULTS_DIR, ensure_dirs
from hermes.research.backtest.frictions import AShareCosts
from hermes.research.backtest.portfolio import signal_portfolio_backtest

INCEPTION = pd.Timestamp(PAPER_INCEPTION)
PRIMARY_TIER = 100_000
SHADOW_DIR = RESULTS_DIR / "paper_shadow" / "d5"
BENCH = RESULTS_DIR / "benchmarks" / "csi300.parquet"
OUT_DIR = RESULTS_DIR / "reviews"
COSTS = AShareCosts()

OUT: dict = {}


def rule(title: str) -> None:
    print(f"\n{'=' * 98}\n{title}\n{'=' * 98}")


def fingerprint(d: Path) -> dict[str, str]:
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def max_dd(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def load_paper(cap: int):
    curve = pd.read_parquet(PAPER_DIR / f"curve_{cap}.parquet")["equity"]
    trades = pd.read_parquet(PAPER_DIR / f"trades_{cap}.parquet")
    report = json.loads((PAPER_DIR / f"report_{cap}.json").read_text(encoding="utf-8"))
    return curve, trades, report


# ----------------------------------------------------------------------------- B1


def b1_integrity(cal: pd.DatetimeIndex) -> dict:
    rule("B1  PAPER INTEGRITY -- operational, reported separately from performance")
    res: dict = {"checks": [], "failures": []}

    def check(ok: bool, label: str, detail: str = "") -> bool:
        res["checks"].append({"label": label, "ok": bool(ok), "detail": detail})
        if not ok:
            res["failures"].append(label)
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  -- ' + detail) if detail else ''}")
        return ok

    expected = cal[cal >= INCEPTION]
    res["expected_bars"] = len(expected)
    print(f"  trading calendar {expected[0].date()} -> {expected[-1].date()}: "
          f"{len(expected)} bars expected\n")

    # D=1 means: execute on the first trading bar of each month after the inception month,
    # plus the inception allocation itself. Derived from the calendar, not typed in.
    months = sorted({(d.year, d.month) for d in expected})
    exp_rebal = [INCEPTION] + [expected[[(x.year, x.month) == m for x in expected]][0]
                               for m in months if m != (INCEPTION.year, INCEPTION.month)]

    for cap in ALL_TIERS:
        curve, trades, report = load_paper(cap)
        tag = f"tier {cap:>9,}"
        check(list(curve.index) == list(expected), f"{tag}: curve covers every expected bar",
              "" if list(curve.index) == list(expected)
              else f"{len(curve)} rows vs {len(expected)} expected")
        check(curve.index.is_monotonic_increasing and not curve.index.has_duplicates,
              f"{tag}: curve index strictly ordered, no duplicate dates")
        check(bool(np.isfinite(curve.to_numpy()).all()) and bool((curve > 0).all()),
              f"{tag}: equity finite and positive throughout")

        # A rebalance that produces no fill is NORMAL at a small tier: the target book may change
        # by less than one lot, or the cash may not cover a lot. So the testable invariant is that
        # the book NEVER trades on a non-D=1 date -- not that every D=1 date has a fill.
        tdates = sorted({pd.Timestamp(d) for d in trades["date"]})
        extra = [d for d in tdates if d not in exp_rebal]
        check(not extra, f"{tag}: trades ONLY on D=1 execution dates",
              f"unexpected trading days {[str(d.date()) for d in extra]}")
        silent = [d for d in exp_rebal if d not in tdates]
        res.setdefault("silent_rebalances", {})[cap] = [str(d.date()) for d in silent]
        if silent:
            print(f"         note: {len(silent)} rebalance(s) produced no fill "
                  f"({[str(d.date()) for d in silent]}) -- sub-lot / unaffordable change, "
                  f"expected at this tier")
        dup = trades.duplicated(subset=["date", "code", "shares", "price"], keep=False)
        check(not dup.any(), f"{tag}: no duplicate fills", f"{int(dup.sum())} duplicated rows")
        check(bool((trades["shares"] != 0).all()), f"{tag}: no zero-share fills")

        folded = (trades.assign(sh=trades["shares"].astype(int))
                  .groupby("code")["sh"].sum())
        folded = {c: int(v) for c, v in folded.items() if v != 0}
        check(folded == {k: int(v) for k, v in report["positions"].items()},
              f"{tag}: trade ledger folds to the reported positions")
        check(bool((pd.Series(folded) > 0).all()), f"{tag}: no negative (short) position")
        check(report["n_trades_total"] == len(trades),
              f"{tag}: report trade count matches the ledger")
        check(abs(float(curve.iloc[-1]) - report["equity"]) < 1e-9,
              f"{tag}: curve endpoint equals the reported equity")
        check(abs(float(curve.iloc[0]) / cap - 1.0) < 0.05,
              f"{tag}: curve starts at the seed capital (post inception-day costs)",
              f"{float(curve.iloc[0]):,.2f} vs seed {cap:,}")

        # Costs: every fill's fee must equal the cost model applied to that fill. The logged
        # `price` is the EXECUTION price (slippage already in it) and `execute_orders` charges the
        # fee on that same post-slippage turnover -- so the fee base is notional as logged, with
        # no de-slipping. (Verified by hand: 100 sh @ 6.633315 -> max(663.33*2.5e-4, 5) +
        # 663.33*1e-5 = 5.006633, exactly the logged fee.)
        notional = (trades["shares"].abs() * trades["price"]).to_numpy()
        side = np.sign(trades["shares"].to_numpy())
        expect_fee = np.array([COSTS.buy_fees(n) if s > 0 else COSTS.sell_fees(n)
                               for n, s in zip(notional, side)])
        gap = float(np.abs(expect_fee - trades["fee"].to_numpy()).max()) if len(trades) else 0.0
        check(gap < 1e-6, f"{tag}: every fee reproduces the cost model", f"max gap {gap:.2e}")

    # idempotency: the production path must reproduce the on-disk ledger exactly
    print()
    for cap in ALL_TIERS:
        _c, _t, report = load_paper(cap)
        rep = live_step(cap, persist=False)
        same = (rep["as_of"] == report["as_of"]
                and abs(rep["equity"] - report["equity"]) < 1e-9
                and rep["positions"] == report["positions"]
                and rep["n_trades_total"] == report["n_trades_total"])
        check(same, f"tier {cap:>9,}: production path reproduces the ledger (idempotent)",
              f"d_equity {rep['equity'] - report['equity']:+.2e}")

    # stale-lake episodes: a run whose data bar lagged the wall clock by more than a long weekend
    logs = sorted((PAPER_DIR / "logs").glob("*.log"))
    stale = []
    for lg in logs:
        txt = lg.read_text(encoding="utf-8", errors="ignore")
        if "STALE" in txt.upper() or "Traceback" in txt:
            stale.append(lg.name)
    res["n_logs"] = len(logs)
    check(not stale, f"no stale-lake or traceback episode in {len(logs)} daily logs",
          f"{stale[:5]}")
    check(report["lake_lag_days"] <= 4 and report["fresh"],
          f"latest run is fresh (lag {report['lake_lag_days']}d)")

    res["expected_rebalances"] = [str(d.date()) for d in exp_rebal]
    print(f"\n  -> {sum(c['ok'] for c in res['checks'])}/{len(res['checks'])} checks passed"
          f"{'' if not res['failures'] else '  FAILURES: ' + str(res['failures'][:6])}")
    return res


# ----------------------------------------------------------------------------- B2


def b2_performance(close: pd.DataFrame) -> dict:
    rule("B2  PERFORMANCE SINCE INCEPTION -- all seven tiers, no cherry-picking")
    rows = []
    for cap in ALL_TIERS:
        curve, trades, report = load_paper(cap)
        notional = float((trades["shares"].abs() * trades["price"]).sum())
        fees = float(trades["fee"].sum())
        last = close.loc[curve.index[-1]]
        pos = {k: int(v) for k, v in report["positions"].items()}
        held_val = float(sum(sh * last.get(c, np.nan) for c, sh in pos.items()))
        cash = float(curve.iloc[-1]) - held_val
        ret = curve / float(curve.iloc[0]) - 1.0
        rows.append({
            "cap": cap, "band": TIER_LABEL[cap], "seed": cap,
            "equity": float(curve.iloc[-1]),
            "total_return": float(report["total_return"]),
            "max_dd": max_dd(curve),
            "vol_ann": float(curve.pct_change().std(ddof=1) * np.sqrt(252)),
            "costs": fees, "costs_bps_of_seed": fees / cap * 1e4,
            "turnover_x": notional / cap, "n_trades": len(trades),
            "n_names": len(pos), "avg_names": float(report["avg_names_held"]),
            "cash": cash, "cash_pct": cash / float(curve.iloc[-1]),
            "ret_series": ret,
        })
    df = pd.DataFrame(rows).set_index("cap")
    print(f"  {'tier':>9} {'band':>7} {'equity':>14} {'net ret':>9} {'maxDD':>8} {'vol':>7} "
          f"{'costs':>10} {'cost bps':>9} {'turn x':>7} {'fills':>6} {'names':>6} {'cash%':>7}")
    for cap in ALL_TIERS:
        r = df.loc[cap]
        print(f"  {cap:>9,} {r['band']:>7} {r['equity']:>14,.2f} {r['total_return']:>+9.2%} "
              f"{r['max_dd']:>8.2%} {r['vol_ann']:>7.1%} {r['costs']:>10,.2f} "
              f"{r['costs_bps_of_seed']:>9.1f} {r['turnover_x']:>7.2f} {int(r['n_trades']):>6} "
              f"{int(r['n_names']):>6} {r['cash_pct']:>7.2%}")

    lo, hi = df["total_return"].min(), df["total_return"].max()
    print(f"\n  cross-tier dispersion: {lo:+.2%} .. {hi:+.2%}  =  {(hi - lo) * 100:.2f} pp spread")
    print("  decomposition of the spread (all four are execution frictions, not signal):")
    print(f"    cost drag          : {df['costs_bps_of_seed'].max() / 100:.2f} pp at 10k vs "
          f"{df['costs_bps_of_seed'].min() / 100:.2f} pp at 5M "
          f"(CNY 5 minimum commission dominates the small tiers)")
    print(f"    names actually held: {int(df['n_names'].min())} at 10k vs "
          f"{int(df['n_names'].max())} at 5M (a 10-name book needs enough capital for 10 lots)")
    print(f"    cash residual      : {df['cash_pct'].max():.2%} at "
          f"{df['cash_pct'].idxmax():,} vs {df['cash_pct'].min():.2%} at "
          f"{df['cash_pct'].idxmin():,} (unfillable lots stay in cash)")
    OUT["tiers"] = {int(c): {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                             for k, v in df.loc[c].items() if k != "ret_series"}
                    for c in ALL_TIERS}
    return {"df": df}


# ----------------------------------------------------------------------------- B3


def b3_benchmark(perf: dict) -> dict:
    rule("B3  BENCHMARK -- CSI 300 TOTAL RETURN (H00300), the index provider's own series")
    bench = pd.read_parquet(BENCH)
    curve = load_paper(PRIMARY_TIER)[0]
    idx = curve.index
    b = bench.reindex(idx).ffill()
    if b.isna().any().any():
        raise SystemExit("ABORT: benchmark does not cover the paper window")
    tr = b["csi300_total_return"]
    px = b["csi300_price"]
    tr_ret = float(tr.iloc[-1] / tr.iloc[0] - 1.0)
    px_ret = float(px.iloc[-1] / px.iloc[0] - 1.0)
    print(f"  window {idx[0].date()} -> {idx[-1].date()}  ({len(idx)} bars)")
    print(f"  CSI300 TOTAL RETURN : {tr_ret:+.2%}")
    print(f"  CSI300 price index  : {px_ret:+.2%}")
    print(f"  dividend contribution inside this window: {(tr_ret - px_ret) * 100:+.2f} pp "
          f"-- the June-July ex-dividend season falls inside it, which is why the price\n"
          f"  index is NOT an acceptable benchmark here and the total-return series is used.")
    rows = []
    for cap in ALL_TIERS:
        h = perf["df"].loc[cap]
        hr = float(h["total_return"])
        rows.append({"cap": cap, "hermes": hr, "bench_tr": tr_ret, "excess_tr": hr - tr_ret,
                     "excess_px": hr - px_ret, "hermes_dd": float(h["max_dd"]),
                     "hermes_vol": float(h["vol_ann"])})
    bdd = max_dd(tr)
    bvol = float(tr.pct_change().std(ddof=1) * np.sqrt(252))
    print(f"\n  {'tier':>9} {'Hermes':>9} {'CSI300TR':>10} {'excess':>9} {'H maxDD':>9} "
          f"{'B maxDD':>9} {'H vol':>7} {'B vol':>7}")
    for r in rows:
        print(f"  {r['cap']:>9,} {r['hermes']:>+9.2%} {tr_ret:>+10.2%} {r['excess_tr']:>+9.2%} "
              f"{r['hermes_dd']:>9.2%} {bdd:>9.2%} {r['hermes_vol']:>7.1%} {bvol:>7.1%}")
    prim = next(r for r in rows if r["cap"] == PRIMARY_TIER)
    print(f"\n  PRIMARY TIER (CNY {PRIMARY_TIER:,}): Hermes {prim['hermes']:+.2%} vs CSI300 TR "
          f"{tr_ret:+.2%}  ->  excess {prim['excess_tr']:+.2%}")
    n = len(idx)
    mech = (1 + prim["hermes"]) ** (252 / n) - 1
    print(f"\n  MECHANICALLY ANNUALISED (statistically weak -- {n} bars, ~1 quarter, do NOT read")
    print(f"  this as an expected return): Hermes {mech:+.1%}/yr, benchmark "
          f"{(1 + tr_ret) ** (252 / n) - 1:+.1%}/yr.")
    dr = load_paper(PRIMARY_TIER)[0].pct_change().dropna()
    hsr = float(dr.mean() / dr.std(ddof=1) * np.sqrt(252)) if len(dr) > 2 else float("nan")
    print(f"  Descriptive Sharpe-like ratio (zero rf, {n} bars): {hsr:.2f} -- reported because it "
          f"was asked for,\n  not as evidence; a one-quarter Sharpe has a standard error of "
          f"roughly +/-1.0 and cannot rank strategies.")
    OUT["benchmark"] = {"bench_tr_return": tr_ret, "bench_px_return": px_ret,
                        "dividend_pp_in_window": (tr_ret - px_ret) * 100,
                        "bench_max_dd": bdd, "bench_vol": bvol,
                        "excess_by_tier": {int(r["cap"]): r["excess_tr"] for r in rows},
                        "mech_annualised_primary": mech}
    return {"tr": tr, "px": px, "rows": rows, "bdd": bdd, "bvol": bvol}


# ----------------------------------------------------------------------------- B4


def b4_context(close: pd.DataFrame, signal: pd.DataFrame, asof, perf: dict) -> dict:
    rule("B4  HISTORICAL CONTEXT -- pre-inception 67-bar windows of the SAME strategy")
    print("  Reference distribution is built ONLY from bars strictly before "
          f"{INCEPTION.date()}.\n  No post-inception data defines it.")
    hist_close = close.loc[close.index < INCEPTION]
    hist_sig = signal.loc[signal.index < INCEPTION]
    n_win = len(perf["df"].loc[PRIMARY_TIER, "ret_series"])
    bench = pd.read_parquet(BENCH)["csi300_total_return"]

    out = {}
    for cap in (PRIMARY_TIER, 1_000_000, 10_000):
        res = signal_portfolio_backtest(hist_close, hist_sig, cap, DEPLOYED.n_hold,
                                        members_asof=asof, weight_asof=DEPLOYED.weight_asof,
                                        rebalance_band=DEPLOYED.rebalance_band,
                                        collect_trades=True)
        eq = res.equity
        tdf = pd.DataFrame(res.trades)
        tdf["notional"] = tdf["shares"].abs() * tdf["price"]
        by_day_not = tdf.groupby("date")["notional"].sum()
        by_day_fee = tdf.groupby("date")["fee"].sum()
        by_day_cnt = tdf.groupby("date").size()
        bt = bench.reindex(eq.index).ffill()

        recs = []
        vals = eq.to_numpy()
        for i in range(len(eq) - n_win + 1):
            w = eq.iloc[i:i + n_win]
            r = vals[i + n_win - 1] / vals[i] - 1.0
            d = w.index
            nt = float(by_day_not.reindex(d).fillna(0).sum())
            fe = float(by_day_fee.reindex(d).fillna(0).sum())
            ct = int(by_day_cnt.reindex(d).fillna(0).sum())
            bw = bt.loc[d]
            br = float(bw.iloc[-1] / bw.iloc[0] - 1.0) if bw.notna().all() else np.nan
            recs.append({"start": d[0], "end": d[-1], "ret": r, "dd": max_dd(w),
                         "vol": float(w.pct_change().std(ddof=1) * np.sqrt(252)),
                         "turnover": nt / float(w.mean()), "cost_bps": fe / float(w.mean()) * 1e4,
                         "n_trades": ct, "excess": r - br})
        H = pd.DataFrame(recs)
        out[cap] = H
        print(f"\n  --- tier CNY {cap:,}: {len(H)} overlapping pre-inception windows of "
              f"{n_win} bars ---")

        curve, trades, _rep = load_paper(cap)
        # TWO DIFFERENT RETURNS, and they must not be mixed silently.
        #   seed-based  = equity / seed - 1        <- the ledger's headline; pays the one-off
        #                                             inception deployment cost
        #   window-based= equity / curve[0] - 1    <- like-for-like with a rolling historical
        #                                             window, which starts already invested
        # The percentile below uses the window-based figure because the reference distribution
        # contains no comparable deployment event; the headline elsewhere is the seed-based one.
        seed_ret = float(curve.iloc[-1] / cap - 1.0)
        fwd_ret = float(curve.iloc[-1] / curve.iloc[0] - 1.0)
        print(f"    return on seed {seed_ret:+.2%} (ledger headline, includes the one-off "
              f"inception deployment cost of {(1 - float(curve.iloc[0]) / cap) * 1e4:.1f} bps);"
              f"\n    return from the first bar {fwd_ret:+.2%} (used for the percentile below, "
              f"like-for-like with rolling windows)")
        fwd_dd = max_dd(curve)
        fwd_vol = float(curve.pct_change().std(ddof=1) * np.sqrt(252))
        bt_f = bench.reindex(curve.index).ffill()
        fwd_ex = fwd_ret - float(bt_f.iloc[-1] / bt_f.iloc[0] - 1.0)
        # The inception bar deploys the whole seed at once; no running 67-bar window contains a
        # comparable event, so turnover and cost are ALSO reported excluding that allocation.
        tr_all = trades.copy()
        tr_all["notional"] = tr_all["shares"].abs() * tr_all["price"]
        post = tr_all[pd.to_datetime(tr_all["date"]) > INCEPTION]
        mean_eq = float(curve.mean())
        fwd_turn_all = float(tr_all["notional"].sum()) / mean_eq
        fwd_turn_post = float(post["notional"].sum()) / mean_eq
        fwd_cost_all = float(tr_all["fee"].sum()) / mean_eq * 1e4
        fwd_cost_post = float(post["fee"].sum()) / mean_eq * 1e4

        def pct(series, v):
            s = series.dropna()
            return float((s < v).mean() * 100) if len(s) else float("nan")

        items = [("return", fwd_ret, H["ret"], "{:+.2%}"),
                 ("max drawdown", fwd_dd, H["dd"], "{:.2%}"),
                 ("volatility", fwd_vol, H["vol"], "{:.1%}"),
                 ("excess vs CSI300TR", fwd_ex, H["excess"], "{:+.2%}"),
                 ("turnover (incl. inception)", fwd_turn_all, H["turnover"], "{:.2f}x"),
                 ("turnover (post-inception)", fwd_turn_post, H["turnover"], "{:.2f}x"),
                 ("cost bps (incl. inception)", fwd_cost_all, H["cost_bps"], "{:.1f}"),
                 ("cost bps (post-inception)", fwd_cost_post, H["cost_bps"], "{:.1f}")]
        print(f"    {'metric':>28} {'forward':>12} {'pctile':>8} {'hist p05':>10} "
              f"{'median':>10} {'p95':>10}")
        prow = {}
        for name, v, s, f in items:
            p = pct(s, v)
            prow[name] = {"forward": v, "percentile": p}
            print(f"    {name:>28} {f.format(v):>12} {p:>7.0f}% {f.format(s.quantile(.05)):>10} "
                  f"{f.format(s.median()):>10} {f.format(s.quantile(.95)):>10}")
        out[f"pct_{cap}"] = prow
    OUT["context"] = {str(k): v for k, v in out.items() if isinstance(k, str)}
    return out


# ----------------------------------------------------------------------------- B5


def b5_attribution(close: pd.DataFrame) -> dict:
    rule("B5  ATTRIBUTION -- what actually moved the book (primary tier)")
    curve, trades, report = load_paper(PRIMARY_TIER)
    idx = curve.index
    px = close.loc[idx]
    trades = trades.copy()
    trades["date"] = pd.to_datetime(trades["date"])

    # Per-name P&L = mark-to-market of the held shares plus realised cash flows, fees included.
    contrib = {}
    for code, g in trades.groupby("code"):
        cash = float(-(g["shares"] * g["price"]).sum() - g["fee"].sum())
        shares = int(g["shares"].sum())
        end_val = shares * float(px[code].iloc[-1]) if shares else 0.0
        contrib[code] = cash + end_val
    s = pd.Series(contrib).sort_values(ascending=False)
    # The base is the SEED, not curve[0]: per-name contributions already carry every fee,
    # including the inception-day deployment cost, so they reconcile to equity - seed exactly.
    total_pnl = float(curve.iloc[-1]) - float(PRIMARY_TIER)
    resid = total_pnl - float(s.sum())
    print(f"  book P&L vs seed: CNY {total_pnl:,.2f} on CNY {PRIMARY_TIER:,} "
          f"({total_pnl / PRIMARY_TIER:+.2%})")
    print(f"  sum of per-name contributions: CNY {s.sum():,.2f}   residual {resid:+.6f} "
          f"-> {'RECONCILES EXACTLY' if abs(resid) < 1e-6 else 'DOES NOT RECONCILE'}")
    if abs(resid) >= 1e-6:
        raise SystemExit("ABORT: attribution does not reconcile to the ledger")
    print(f"\n  {'top contributors':>16} {'CNY':>12} {'pp of seed':>11}   "
          f"{'worst contributors':>18} {'CNY':>12} {'pp of seed':>11}")
    top, bot = s.head(5), s.tail(5).iloc[::-1]
    for (ct, vt), (cb, vb) in zip(top.items(), bot.items()):
        print(f"  {ct:>16} {vt:>12,.0f} {vt / PRIMARY_TIER * 100:>10.2f}p   "
              f"{cb:>18} {vb:>12,.0f} {vb / PRIMARY_TIER * 100:>10.2f}p")
    n_pos = int((s > 0).sum())
    gross_up, gross_dn = float(s[s > 0].sum()), float(s[s < 0].sum())
    print(f"\n  {n_pos} of {len(s)} names positive. Gross winners CNY {gross_up:,.0f} "
          f"({gross_up / PRIMARY_TIER * 100:+.2f}pp), gross losers CNY {gross_dn:,.0f} "
          f"({gross_dn / PRIMARY_TIER * 100:+.2f}pp).")
    print(f"  The NET is a small difference of two much larger gross numbers, so ratios of any "
          f"single name\n  to the net ({total_pnl:,.0f}) are unstable and are deliberately not "
          f"quoted. Best name {s.index[0]} = {s.iloc[0] / PRIMARY_TIER * 100:+.2f}pp of seed.")
    concentration = float(s.abs().sort_values(ascending=False).head(3).sum() / s.abs().sum())
    print(f"  top-3 names carry {concentration:.0%} of the ABSOLUTE contribution -- "
          f"{'concentrated' if concentration > 0.5 else 'broadly distributed'}")

    print("\n  per-rebalance book return (D=1 execution dates):")
    rebal = sorted({pd.Timestamp(d) for d in trades['date']})
    bounds = rebal + [idx[-1]]
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = curve.loc[a:b]
        if len(seg) > 1:
            print(f"    {a.date()} -> {b.date()}  {len(seg) - 1:>3} bars  "
                  f"{float(seg.iloc[-1] / seg.iloc[0] - 1):>+8.2%}")
    OUT["attribution"] = {"total_pnl": total_pnl, "top": {k: float(v) for k, v in top.items()},
                          "bottom": {k: float(v) for k, v in bot.items()},
                          "n_positive": n_pos, "n_names": int(len(s)),
                          "top3_abs_share": concentration}
    return {"contrib": s}


# ----------------------------------------------------------------------------- B6


def b6_shadow() -> dict:
    rule("B6  D=5 FORWARD SHADOW (issue #22) -- separate evidence, NOT part of the D=1 review")
    man = json.loads((SHADOW_DIR / "manifest.json").read_text(encoding="utf-8"))
    print(f"  shadow inception {man['shadow_inception_asof']}  candidate D={man['candidate_calendar_day']} "
          f"vs baseline D={man['baseline_calendar_day']}")
    rows = []
    for cap in ALL_TIERS:
        r = json.loads((SHADOW_DIR / f"report_{cap}.json").read_text(encoding="utf-8"))
        rows.append(r)
        print(f"  {cap:>9,}: as_of {r['as_of']}  D5 {r['total_return_since_shadow_inception']:>+7.3%}"
              f"  D1 {r['d1_return_since_shadow_inception']:>+7.3%}"
              f"  excess {r['excess_return_vs_d1']:>+7.3%}"
              f"  ddiff {r.get('drawdown_diff_vs_d1', float('nan')):>+7.3%}"
              f"  cdiff {r.get('cost_diff_vs_d1', float('nan')):>9,.2f}"
              f"  fills {r['n_trades_since_shadow_inception']:>3}")
    cal = pd.read_parquet(PAPER_DIR / f"curve_{PRIMARY_TIER}.parquet").index
    post = cal[cal > pd.Timestamp(man["shadow_inception_asof"])]
    d1_ex = [d for d in post if d.day <= 3 and d == post[[x.year == d.year and x.month == d.month
                                                          for x in post]][0]]
    print(f"\n  bars since fork: {len(post)}    divergent rebalance events so far: "
          f"D=1 traded {len(d1_ex)} time(s) post-fork, the shadow traded on its own D=5 dates.")
    rep = json.loads((SHADOW_DIR / f"report_{PRIMARY_TIER}.json").read_text(encoding="utf-8"))
    dp = set(rep["positions"]) ^ set(rep["d1_positions"])
    print(f"  primary tier position difference: {len(dp)} name(s) differ -> {sorted(dp)}")
    print(f"\n  {len(post)} bars is FAR too short to mean anything. Issue #22 freezes the primary")
    print("  verdict at 24 complete comparable monthly cycles; #21 and #23 already concluded that")
    print("  D=5's historical edge does not justify replacing D=1. Nothing here changes that.")
    OUT["shadow"] = {"bars_since_fork": int(len(post)),
                     "excess_by_tier": {int(c): json.loads(
                         (SHADOW_DIR / f"report_{c}.json").read_text(encoding="utf-8")
                     )["excess_return_vs_d1"] for c in ALL_TIERS}}
    return {"rows": rows}


# ----------------------------------------------------------------------------- main


def main() -> None:
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    before = fingerprint(PAPER_DIR)

    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof = membership_lookup(mdf)
    close = load_close_panel(codes=union, field="close")
    pe = load_close_panel(codes=union, field="peTTM")
    signal = deployed_signal(close, pe, asof)
    cal = close.index

    print(f"canonical source: {PAPER_DIR}")
    print(f"lake latest bar : {cal[-1].date()}   inception: {INCEPTION.date()}")

    integrity = b1_integrity(cal)
    perf = b2_performance(close)
    bench = b3_benchmark(perf)
    b4_context(close, signal, asof, perf)
    b5_attribution(close)
    b6_shadow()

    after = fingerprint(PAPER_DIR)
    rule("GOVERNANCE")
    same = before == after
    print(f"  results/paper/ unchanged by this review: {'PASS' if same else 'FAIL'} "
          f"({len(before)} files, SHA256)")
    if not same:
        raise SystemExit("ABORT: this review modified the canonical record")
    print("  DEPLOYED, PAPER_INCEPTION, the D=1 schedule and the D=5 candidate: not touched.")

    OUT["meta"] = {"as_of": str(cal[-1].date()), "inception": str(INCEPTION.date()),
                   "bars": int(integrity["expected_bars"]),
                   "integrity_pass": int(sum(c["ok"] for c in integrity["checks"])),
                   "integrity_total": int(len(integrity["checks"])),
                   "integrity_failures": integrity["failures"],
                   "rebalances": integrity["expected_rebalances"],
                   "bench_tr": bench["rows"][0]["bench_tr"]}
    (OUT_DIR / "forward_review.json").write_text(json.dumps(OUT, indent=2, default=float),
                                                 encoding="utf-8")
    print(f"\nwrote {OUT_DIR / 'forward_review.json'}")


if __name__ == "__main__":
    main()
