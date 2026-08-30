"""Calendar-anchor rebalance study, supplement to issue #21. Issue #23.

Issue #21 already swept D=1..31 and closed REJECTED / NOT IDENTIFIED. This does NOT repeat it. It
closes four gaps found by re-reading the specification against what #21 froze and delivered:

  1  SEMANTICS  #21 clamped execution to the anchor's own calendar month; here roll-forward across
                the month boundary is allowed (the requested definition). Identical for D=1..23 on
                this lake, different for D=24..31.
  2  WINDOW     a genuine trailing decade, 2016-08-29..2026-08-28, which #21 never evaluated and
                which contains eight months of data that did not exist when #21 ran.
  3  ROLLING    #21 pre-registered rolling-window stability and never implemented it. Delivered here.
  4  ATTRIBUTION why the days differ -- basket overlap, names changed, signal-rank movement, and
                whether a few extreme months carry a delta. #21 never asked.

Primary ranking basis: CNY 100,000 net of frictions. Robustness at 30k and 500k. Everything except
the anchor day is the deployed spec, unchanged; no parameter is retuned jointly with the anchor.

    python scripts/rebalance_anchor_day_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.live.strategy import DEPLOYED, deployed_signal
from hermes.paths import FIGURES_DIR, RESULTS_DIR
from hermes.research.backtest.frictions import ZERO_COSTS
from hermes.research.backtest.portfolio import signal_portfolio_backtest
from hermes.research.backtest.schedule import (anchor_rollforward_schedule, assert_no_lookahead,
                                               month_end_schedule, schedule_audit_rows,
                                               turnover_from_trades)

DAYS = list(range(1, 32))
BASELINE = 1
PRIMARY_TIER = 100_000
ROBUST_TIERS = [30_000, 500_000]
WINDOW_YEARS = 10
TRADING_DAYS = 243
ROLL_YEARS = 3
ROLL_STEP_MONTHS = 6
OUT = RESULTS_DIR / "studies" / "rebalance_anchor_day"


# --- metrics -------------------------------------------------------------------------

def metrics(res, trades_in_window=None) -> dict:
    eq = res.equity
    r = eq.pct_change().dropna()
    sd = float(r.std(ddof=1))
    dd = res.max_drawdown
    tr = res.trades if trades_in_window is None else trades_in_window
    return {"n_rebalances": res.n_rebalances, "total_return": res.total_return, "cagr": res.cagr,
            "ann_vol": sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan,
            "max_drawdown": dd, "calmar": res.cagr / abs(dd) if dd else np.nan,
            "sharpe": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
            "total_costs": res.total_costs, "turnover": turnover_from_trades(tr),
            "avg_names_held": res.avg_names_held, "terminal_equity": float(eq.iloc[-1])}


def window_cagr(eq: pd.Series, lo, hi) -> float:
    seg = eq.loc[lo:hi]
    if len(seg) < 2:
        return np.nan
    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
    return float((seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1.0) if yrs > 0 else np.nan


def window_calmar(eq: pd.Series, lo, hi) -> float:
    seg = eq.loc[lo:hi]
    if len(seg) < 2:
        return np.nan
    dd = float((seg / seg.cummax() - 1.0).min())
    c = window_cagr(eq, lo, hi)
    return c / abs(dd) if dd else np.nan


def window_return(eq: pd.Series, lo, hi) -> float:
    """Plain period return over a window -- NOT annualised.

    The evaluation window is a trailing decade, so its first and last calendar years are stubs
    (2016 spans ~0.34 yr, 2026 ~0.64 yr). Annualising those cubes the first one, which would put a
    distorted number in the annual table and then feed it into the drop-one-year test."""
    seg = eq.loc[lo:hi]
    return float(seg.iloc[-1] / seg.iloc[0] - 1.0) if len(seg) >= 2 else np.nan


def holdings_by_bar(trades, dates: pd.DatetimeIndex) -> dict:
    """Fold the fill log into the HELD book at each bar.

    The engine's trade log is fills, not holdings: it contains names being sold OUT (which are by
    definition not in the resulting basket) and omits held names whose lot-rounded target happened
    to equal their current shares. Overlap between two schedules must be measured on what is HELD.
    """
    by_day = {}
    for tr in trades:
        by_day.setdefault(tr["date"], []).append(tr)
    pos, out = {}, {}
    for d in dates:
        for tr in by_day.get(d, ()):
            pos[tr["code"]] = pos.get(tr["code"], 0) + int(tr["shares"])
        out[d] = {c for c, s in pos.items() if s > 0}
    return out


def run(close, signal, cap, asof, sched, costs=None):
    return signal_portfolio_backtest(close, signal, cap, DEPLOYED.n_hold, costs=costs,
                                     members_asof=asof, rebalance_band=DEPLOYED.rebalance_band,
                                     collect_trades=True, schedule=sched)


def sweep(close, signal, asof, cap, gross=False):
    rows, curves, baskets = [], {}, {}
    for d in DAYS:
        sch = anchor_rollforward_schedule(close.index, d)
        assert_no_lookahead(sch, close.index)
        res = run(close, signal, cap, asof, sch, ZERO_COSTS if gross else None)
        audit = schedule_audit_rows(close.index, d, sch)
        curves[d] = res.equity
        baskets[d] = res.trades
        rows.append({"D": d, **metrics(res),
                     "rolled_pct": float(np.mean([a["rolled_for_weekend_or_holiday"] for a in audit]))
                     if audit else np.nan,
                     "crossed_month": int(sum(a["crossed_calendar_month"] for a in audit))})
    df = pd.DataFrame(rows).set_index("D")
    for c, nm in [("cagr", "dCAGR"), ("calmar", "dCalmar"), ("sharpe", "dSharpe"),
                  ("max_drawdown", "dMaxDD"), ("total_costs", "dCosts")]:
        df[nm] = df[c] - df.loc[BASELINE, c]
    return df, curves, baskets


def fmt(df, cols, title, pct=(), num=()):
    d = df[cols].copy()
    for c in pct:
        if c in d:
            d[c] = d[c].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "--")
    for c in num:
        if c in d:
            d[c] = d[c].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "--")
    print(f"\n{title}")
    print(d.to_string())


PCT = ("total_return", "cagr", "ann_vol", "max_drawdown", "dCAGR", "dMaxDD", "rolled_pct")
NUM = ("calmar", "sharpe", "dCalmar", "dSharpe")


def main() -> None:
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof = membership_lookup(mdf)
    close_all = load_close_panel(codes=union, field="close")
    pe_all = load_close_panel(codes=union, field="peTTM")
    signal_all = deployed_signal(close_all, pe_all, asof, DEPLOYED)   # warm lookbacks on full history

    end = close_all.index[-1]
    start = close_all.index[close_all.index >= (end - pd.DateOffset(years=WINDOW_YEARS))][0]
    close = close_all.loc[start:end]
    signal = signal_all.loc[start:end]
    print("=== window ===")
    print(f"  lake ends {end.date()}; evaluation {start.date()} .. {end.date()} "
          f"({len(close.index)} bars, {len({(d.year, d.month) for d in close.index})} months)")
    print("  signals computed on the FULL history then sliced -- no candidate pays a warm-up penalty")

    # --- BLOCKING PARITY GATE ---------------------------------------------------
    print("\n=== D=1 PARITY GATE (blocking) ===")
    s1 = anchor_rollforward_schedule(close.index, BASELINE)
    legacy = month_end_schedule(close.index)
    assert s1 == legacy, "D=1 roll-forward schedule != legacy monthly schedule"
    a = run(close, signal, PRIMARY_TIER, asof, None)
    b = run(close, signal, PRIMARY_TIER, asof, s1)
    assert a.equity.equals(b.equity), "D=1 equity differs from the legacy engine"
    assert len(a.trades) == len(b.trades) and a.n_rebalances == b.n_rebalances
    print(f"  schedules identical: True   rebalances {a.n_rebalances}   trades {len(a.trades)}")
    print(f"  equity max abs diff {float((a.equity - b.equity).abs().max()):.3e}   "
          f"costs {a.total_costs:,.2f} vs {b.total_costs:,.2f}   -> PASS")

    # --- PRIMARY SWEEP ----------------------------------------------------------
    net, curves, baskets = sweep(close, signal, asof, PRIMARY_TIER)
    gross, _cg, _bg = sweep(close, signal, asof, PRIMARY_TIER, gross=True)

    fmt(net, ["n_rebalances", "total_return", "cagr", "ann_vol", "max_drawdown", "calmar",
              "sharpe", "total_costs", "turnover", "avg_names_held", "rolled_pct", "crossed_month"],
        f"=== A. all 31 anchors, calendar order, NET (CNY {PRIMARY_TIER:,}) ===", PCT, NUM)
    fmt(net.sort_values("cagr", ascending=False),
        ["cagr", "dCAGR", "calmar", "sharpe", "max_drawdown", "total_costs"],
        "=== B1. ranked by CAGR (all 31) ===", PCT, NUM)
    fmt(net.sort_values("calmar", ascending=False),
        ["calmar", "dCalmar", "cagr", "max_drawdown", "ann_vol"],
        "=== B2. ranked by Calmar (all 31) ===", PCT, NUM)
    print(f"\n=== C/D. baseline D=1: CAGR {net.loc[1, 'cagr']:+.2%}, "
          f"Calmar {net.loc[1, 'calmar']:.3f}, rank by CAGR "
          f"{int(net['cagr'].rank(ascending=False)[1])}/31 ===")
    fmt(gross, ["cagr", "calmar", "max_drawdown"],
        "=== secondary diagnostic: GROSS (zero cost) -- gross effect or turnover artefact? ===",
        PCT, NUM)

    # --- 9.1 chronological split ------------------------------------------------
    mid = close.index[len(close.index) // 2]
    disc = {d: window_cagr(curves[d], start, mid) for d in DAYS}
    vald = {d: window_cagr(curves[d], mid, end) for d in DAYS}
    dr = pd.Series(disc).rank(ascending=False)
    vr = pd.Series(vald).rank(ascending=False)
    split = pd.DataFrame({"discovery_cagr": disc, "discovery_rank": dr,
                          "validation_cagr": vald, "validation_rank": vr})
    print(f"\n=== 9.1 chronological split at {mid.date()} ===")
    print(split.sort_values("discovery_rank").head(10).to_string(
        formatters={"discovery_cagr": "{:+.2%}".format, "validation_cagr": "{:+.2%}".format}))
    dw = int(pd.Series(disc).idxmax())
    rho = float(dr.corr(vr, method="spearman"))
    print(f"\n  discovery winner D={dw}: discovery {disc[dw]:+.2%} (rank 1) -> "
          f"validation {vald[dw]:+.2%} (rank {int(vr[dw])}/31)")
    for k in (3, 5):
        ov = len(set(dr.nsmallest(k).index) & set(vr.nsmallest(k).index))
        print(f"  top-{k} overlap across halves: {ov}/{k}")
    print(f"  rank correlation across halves: rho {rho:+.3f}")

    # --- 9.2 rolling stability (issue #21 promised this and never delivered it) ---
    wins = []
    t = start
    while t + pd.DateOffset(years=ROLL_YEARS) <= end:
        wins.append((t, t + pd.DateOffset(years=ROLL_YEARS)))
        t = t + pd.DateOffset(months=ROLL_STEP_MONTHS)
    ranks, beats, deltas = {}, {}, {}
    for d in DAYS:
        rs, bs, ds = [], [], []
        for lo, hi in wins:
            vals = {k: window_cagr(curves[k], lo, hi) for k in DAYS}
            s = pd.Series(vals).rank(ascending=False)
            rs.append(s[d])
            ds.append(vals[d] - vals[BASELINE])
            bs.append(vals[d] > vals[BASELINE])
        ranks[d], beats[d], deltas[d] = rs, np.mean(bs), ds
    roll = pd.DataFrame({"median_rank": {d: float(np.median(ranks[d])) for d in DAYS},
                         "worst_rank": {d: float(np.max(ranks[d])) for d in DAYS},
                         "share_beating_D1": beats,
                         "median_dCAGR": {d: float(np.median(deltas[d])) for d in DAYS}})
    roll.index.name = "D"
    fmt(roll.sort_values("median_rank"), list(roll.columns),
        f"=== 9.2 rolling stability, {ROLL_YEARS}y windows stepped {ROLL_STEP_MONTHS}m "
        f"({len(wins)} windows) ===", ("share_beating_D1", "median_dCAGR"))

    # --- 9.3 neighbours ---------------------------------------------------------
    champ = int(net["cagr"].idxmax())
    print(f"\n=== 9.3 neighbour context around the highest-CAGR candidate D={champ} ===")
    for d in range(max(1, champ - 3), min(31, champ + 3) + 1):
        mark = "  <- champion" if d == champ else ""
        print(f"  D={d:<3} CAGR {net.loc[d, 'cagr']:+.2%}  dCAGR {net.loc[d, 'dCAGR']:+.2%}"
              f"  Calmar {net.loc[d, 'calmar']:.3f}{mark}")

    # --- 9.4 annual ------------------------------------------------------------
    # PLAIN calendar-year returns, not annualised: the window's first and last years are stubs
    # (2016 ~0.34 yr, 2026 ~0.64 yr) and annualising would cube the first one.
    yrs = sorted({d.year for d in close.index})
    ann = pd.DataFrame({d: {y: window_return(curves[d], f"{y}-01-01", f"{y}-12-31") for y in yrs}
                        for d in DAYS}).T
    ann.index.name = "D"
    rel = ann.sub(ann.loc[BASELINE], axis=1)
    partial = [y for y in yrs
               if len(close.index[(close.index >= f"{y}-01-01") & (close.index <= f"{y}-12-31")]) < 200]
    print("\n=== 9.4 calendar-year return vs D=1 (top-5 by full-window CAGR; NOT annualised) ===")
    top5 = net["cagr"].nlargest(5).index.tolist()
    print(rel.loc[top5].to_string(float_format=lambda v: f"{v:+.1%}"))
    if partial:
        print(f"  note: {partial} are PARTIAL years at the window edges -- shorter, not weaker")

    # Drop-one-year: recompute the COMPOUNDED full-window edge with that year's months removed,
    # which is what "would the edge survive without this year" actually means. An arithmetic mean
    # of per-year numbers is neither the full-window figure nor equally weighted in time.
    mret = {d: curves[d].resample("ME").last().pct_change().dropna() for d in (BASELINE, champ)}
    def _compounded(d, drop_year=None):
        s = mret[d]
        if drop_year is not None:
            s = s[[ts.year != drop_year for ts in s.index]]
        return float((1.0 + s).prod() - 1.0)
    base_edge = _compounded(champ) - _compounded(BASELINE)
    print(f"\n  drop-one-year test (compounded full-window edge vs D=1; all years {base_edge:+.2%}):")
    for y in yrs:
        e = _compounded(champ, y) - _compounded(BASELINE, y)
        flag = "  <- partial year" if y in partial else ""
        print(f"    excluding {y}: {e:+.2%}   (change {e - base_edge:+.2%}){flag}")

    # --- 14. economic attribution ----------------------------------------------
    print(f"\n=== 14. economic attribution, D={champ} vs D=1 ===")
    e1 = set(pd.to_datetime([tr["date"] for tr in baskets[BASELINE]]))
    ec = set(pd.to_datetime([tr["date"] for tr in baskets[champ]]))
    print(f"  execution dates: D=1 {len(e1)} distinct, D={champ} {len(ec)}, shared {len(e1 & ec)}")

    # Overlap of what is HELD, on a common grid of every bar -- NOT of fills on the dates where the
    # two schedules happen to execute together. Those shared dates are exactly the long-holiday
    # months where the anchors collapse onto one bar, i.e. the subsample in which the schedules do
    # NOT differ; conditioning on them measures agreement and calls it attribution. For most
    # candidates that intersection is empty and the line silently vanished.
    h1 = holdings_by_bar(baskets[BASELINE], close.index)
    hc = holdings_by_bar(baskets[champ], close.index)
    jac = [len(h1[d] & hc[d]) / len(h1[d] | hc[d]) for d in close.index if (h1[d] | hc[d])]
    changed = [len(h1[d] ^ hc[d]) for d in close.index if (h1[d] | hc[d])]
    print(f"  holdings Jaccard over all {len(jac)} bars: median {np.median(jac):.2f}, "
          f"mean {np.mean(jac):.2f}, min {np.min(jac):.2f}")
    print(f"  names differing per bar: median {np.median(changed):.1f}, max {int(np.max(changed))}")
    print(f"  turnover  D=1 {net.loc[1, 'turnover']:,.0f}  D={champ} {net.loc[champ, 'turnover']:,.0f}")
    print(f"  costs     D=1 {net.loc[1, 'total_costs']:,.0f}  D={champ} {net.loc[champ, 'total_costs']:,.0f}")

    # "Do a few extreme months carry the delta?" -- rank by ABSOLUTE delta. Taking the 3 most
    # negative plus the 3 most positive would net them against each other by construction, and the
    # reference must be the COMPOUNDED terminal gap, not an arithmetic sum of monthly differences.
    m1 = curves[BASELINE].resample("ME").last().pct_change().dropna()
    mc = curves[champ].resample("ME").last().pct_change().dropna()
    dm = (mc - m1).dropna()
    gap = float((1.0 + mc).prod() - (1.0 + m1).prod())
    top6 = dm.reindex(dm.abs().sort_values(ascending=False).index[:6])
    print(f"\n  monthly deltas over {len(dm)} months; compounded terminal gap {gap:+.2%}")
    print(f"  the 6 largest-|delta| months sum to {top6.sum():+.2%}"
          + (f" = {top6.sum() / gap * 100:+.0f}% of the terminal gap" if abs(gap) > 1e-6 else ""))
    for k, v in top6.sort_values().items():
        print(f"    {k.strftime('%Y-%m')}: {v:+.2%}")
    without = dm.drop(top6.index)
    print(f"  the other {len(without)} months sum to {without.sum():+.2%} "
          f"-> the edge is {'concentrated in a few months' if abs(top6.sum()) > abs(without.sum()) else 'spread across many months'}")

    # --- capital robustness ----------------------------------------------------
    tiers = {}
    for cap in ROBUST_TIERS:
        t_df, _c, _b = sweep(close, signal, asof, cap)
        tiers[cap] = t_df
        fmt(t_df.sort_values("cagr", ascending=False).head(8),
            ["cagr", "dCAGR", "calmar", "max_drawdown"],
            f"=== capital robustness CNY {cap:,} (top 8; ranking basis stays {PRIMARY_TIER:,}) ===",
            PCT, NUM)

    _verdict(net, gross, split, roll, tiers, champ, rho, curves, len(wins))
    _persist(net, gross, split, roll, ann, close, curves)


def _verdict(net, gross, split, roll, tiers, champ, rho, curves, n_windows) -> None:
    print("\n" + "=" * 78)
    print("=== VERDICT (classification frozen in issue #23) ===")
    spread = net["cagr"].max() - net["cagr"].min()
    print(f"\n  highest-CAGR candidate on this window: D={champ} ({net.loc[champ, 'cagr']:+.2%}); "
          f"worst D={int(net['cagr'].idxmin())} ({net['cagr'].min():+.2%}); spread {spread:.2%}")
    print("  (descriptive only -- 'highest on this window', never 'optimal')")
    nb = [d for d in range(max(1, champ - 2), min(31, champ + 2) + 1)
          if d != BASELINE and net.loc[d, "dCAGR"] > 0]
    plateau = len(nb) >= 3
    val_rank = int(split.loc[champ, "validation_rank"])
    validates = val_rank <= 8
    tiers_ok = all(t.loc[champ, "dCAGR"] > 0 for t in tiers.values())
    rolls_ok = roll.loc[champ, "share_beating_D1"] >= 0.6
    gross_ok = gross.loc[champ, "cagr"] > gross.loc[BASELINE, "cagr"]
    # Section 9.4, which the branch previously printed and then ignored: does the edge survive
    # removing the one or two years that carry it? Operationalised as the COMPOUNDED edge with the
    # two largest positive annual contributors dropped -- if that turns negative, the full-sample
    # result is two good years wearing a plateau's clothes.
    m1 = curves[BASELINE].resample("ME").last().pct_change().dropna()
    mc = curves[champ].resample("ME").last().pct_change().dropna()
    idx = m1.index.intersection(mc.index)
    m1, mc = m1[idx], mc[idx]

    def _edge(drop=()):
        k = [ts for ts in idx if ts.year not in drop]
        return float((1.0 + mc[k]).prod() - (1.0 + m1[k]).prod())

    years = sorted({ts.year for ts in idx})
    contrib = sorted(years, key=lambda y: _edge() - _edge((y,)), reverse=True)
    worst2 = tuple(contrib[:2])
    edge_all, edge_ex2 = _edge(), _edge(worst2)
    year_robust = edge_ex2 > 0

    for name, ok in [("neighbours support it (plateau >= 3 of +-2)", plateau),
                     ("validation half rank <= 8", validates),
                     (f"rolling: beats D=1 in >= 60% of windows ({roll.loc[champ, 'share_beating_D1']:.0%})",
                      rolls_ok),
                     ("gross agrees (not a turnover artefact)", gross_ok),
                     ("all capital tiers agree", tiers_ok),
                     (f"edge survives dropping its two best years {worst2}: "
                      f"{edge_all:+.2%} -> {edge_ex2:+.2%}", year_robust)]:
        print(f"  [{'yes' if ok else 'no '}] {name}")
    print(f"\n  caveat on the rolling figure: {n_windows} windows of {ROLL_YEARS}y stepped "
          f"{ROLL_STEP_MONTHS}m over a {WINDOW_YEARS}y span OVERLAP heavily -- at most "
          f"{WINDOW_YEARS // ROLL_YEARS} disjoint windows exist, so a high share is not that many "
          f"independent observations and inherits any dominant year repeatedly.")
    # Frozen definitions (issue #23). C is "dominates the full sample but fails the split/rolling
    # tests" -- an earlier ordering sent exactly that truth assignment (no plateau, fails both OOS
    # checks) into the A branch, mislabelling the textbook isolated winner as no effect. The
    # discriminator between A and C is whether the candidate DOMINATES the full sample at all.
    # Year robustness is a NECESSARY condition for B and D: a candidate whose advantage vanishes
    # when its two best years are removed has not shown a repeated monthly edge, whatever the
    # overlapping rolling windows suggest.
    oos_ok = validates and rolls_ok and year_robust
    dominates = bool(net.loc[champ, "dCAGR"] > 0.01 and spread > 0.02)   # a real full-sample gap
    if oos_ok and tiers_ok and plateau:
        v = "B. BROAD ROBUST WINDOW -- a neighbouring group holds up; candidate for a forward shadow test"
    elif oos_ok and tiers_ok:
        v = "D. ROBUST SPECIFIC ANCHOR -- holds up without neighbour support; forward test, NOT proven alpha"
    elif dominates:
        v = ("C. ISOLATED HISTORICAL WINNER -- dominates the full sample but fails "
             + ("the plateau test" if not plateau else "")
             + (" and " if not plateau and not oos_ok else "")
             + ("the out-of-sample tests" if not oos_ok else "")
             + (f" (its edge is {edge_all:+.1%} but {edge_ex2:+.1%} without {worst2})"
                if not year_robust else "")
             + "; likely selection noise, do not deploy")
    else:
        v = "A. NO EFFECT -- no meaningful full-sample separation, ranks unstable; keep D=1"
    print(f"\n  VERDICT: {v}")
    print(f"  cross-half rank correlation rho {rho:+.3f}")
    print("\n  DEPLOYED, PAPER_INCEPTION and results/paper are unchanged by this study (issue #23).")
    print("=" * 78)


def _persist(net, gross, split, roll, ann, close, curves) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    net.to_csv(OUT / "summary_by_day.csv")
    gross.to_csv(OUT / "summary_by_day_gross.csv")
    split.to_csv(OUT / "chronological_split.csv")
    roll.to_csv(OUT / "rolling_stability.csv")
    ann.to_csv(OUT / "annual_returns_by_day.csv")
    rows = []
    for d in DAYS:
        rows += schedule_audit_rows(close.index, d, anchor_rollforward_schedule(close.index, d))
    pd.DataFrame(rows).to_csv(OUT / "schedule.csv", index=False)
    pd.DataFrame({d: curves[d].resample("ME").last().pct_change() for d in DAYS}).to_csv(
        OUT / "monthly_returns_by_day.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(net.index, net["cagr"] * 100, "o-", label="net")
    ax[0, 0].plot(gross.index, gross["cagr"] * 100, "s--", alpha=.6, label="gross")
    ax[0, 0].axhline(net.loc[BASELINE, "cagr"] * 100, color="r", ls=":", label="D=1")
    ax[0, 0].set_title("CAGR by calendar anchor")
    ax[0, 0].set_xlabel("anchor day")
    ax[0, 0].legend()
    ax[0, 1].plot(net.index, net["calmar"], "o-")
    ax[0, 1].axhline(net.loc[BASELINE, "calmar"], color="r", ls=":")
    ax[0, 1].set_title("Calmar by calendar anchor")
    ax[0, 1].set_xlabel("anchor day")
    ax[1, 0].plot(split.index, split["discovery_rank"], "o-", label="discovery half")
    ax[1, 0].plot(split.index, split["validation_rank"], "s-", label="validation half")
    ax[1, 0].set_title("rank by half (1 = best)")
    ax[1, 0].set_xlabel("anchor day")
    ax[1, 0].legend()
    ax[1, 1].plot(roll.index, roll["share_beating_D1"] * 100, "o-")
    ax[1, 1].axhline(50, color="k", lw=.8)
    ax[1, 1].set_title(f"share of rolling {ROLL_YEARS}y windows beating D=1 (%)")
    ax[1, 1].set_xlabel("anchor day")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "rebalance_anchor_day.png", dpi=130)
    plt.close(fig)
    print(f"\nsaved -> {OUT}\nfigure -> {FIGURES_DIR / 'rebalance_anchor_day.png'}")


if __name__ == "__main__":
    main()
