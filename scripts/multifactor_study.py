"""Step B: multi-factor diversification, to add BREADTH to the value factor (Calmar 0.28).

The discipline: prove diversification exists BEFORE combining, and SWEEP the mix densely
(an earlier version sampled only 3 heavy-reversal weights, landed in the worst region of
the curve, and wrongly concluded "no combo helps" -- adversarial review caught it). Here we
show the whole curve and report NET and GROSS (zero-cost) Calmar: if GROSS rises as reversal
is added, the gain is genuine pre-cost alpha, not luck surviving frictions.

All results: survivorship-free PIT HS300, A-share frictions, top-10 monthly, 2015-2025.
Every cross-sectional blend input is restrict_to_universe()'d to the then-current members
BEFORE standardization (else the union leaks -- see A2 / risk_control.md).

    python scripts/multifactor_study.py
"""
import pandas as pd

from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.paths import BACKTESTS_DIR
from hermes.research.backtest.frictions import ZERO_COSTS
from hermes.research.backtest.portfolio import signal_portfolio_backtest
from hermes.research.backtest.sizing import inverse_vol_weighter
from hermes.research.eval.factor_eval import compute_ic
from hermes.research.factors import library as fl

TIERS = [100_000, 1_000_000, 10_000_000]
N_HOLD = 10


def main() -> None:
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof = membership_lookup(mdf)

    close = load_close_panel(codes=union, field="close")
    ep = fl.earnings_yield(load_close_panel(codes=union, field="peTTM"))
    bp = fl.book_yield(load_close_panel(codes=union, field="pbMRQ"))
    ps = load_close_panel(codes=union, field="psTTM")
    sp = (1.0 / ps).where(ps > 0)
    lowvol = fl.low_vol(close, 120)
    rev1 = -fl.trailing_return(close, 20)       # 1-month reversal
    rev10 = -fl.trailing_return(close, 10)      # 2-week reversal (shorter)
    mom = fl.momentum(close, 120, 20)           # 6-1m momentum
    eval_dates = [pd.Timestamp(d) for d in sorted(mdf["date"].unique()) if pd.Timestamp(d) in close.index]

    # 1) DIAGNOSIS: monthly rank-IC correlation. A useful FIRST screen (it flags the value
    #    family + low-vol as mutually redundant and reversal as the lone negative), but it is
    #    window-sensitive and lives in rank space, not return space -- so it under-ranks the
    #    diversifiers (a 10-day reversal shows ~0 IC-corr yet combines best). The real proof
    #    is the combination curve in part 2, not this matrix.
    factors = {"ep": ep, "bp": bp, "sp": sp, "lowvol": lowvol, "rev1m": rev1, "mom6_1": mom}
    ic_series = {k: compute_ic(f, close, eval_dates, members_asof=asof).ic for k, f in factors.items()}
    print("1) factor monthly rank-IC correlation (screen only; see caveat in source):")
    print(pd.DataFrame(ic_series).corr().round(2).to_string())
    print("   => value family (ep/bp/sp) + low-vol mutually 0.7-0.9 (redundant); reversal the\n"
          "      only negative. Confirms A2's value x low-vol null from the factor side.\n")

    ep_pit, rev1_pit, rev10_pit = (fl.restrict_to_universe(x, asof) for x in (ep, rev1, rev10))
    invv = inverse_vol_weighter(close, lookback=60)

    def cal(r):
        return r.cagr / abs(r.max_drawdown) if r.max_drawdown < 0 else float("nan")

    def bt(sig, cap, costs=None, wfn=None, band=0):
        return signal_portfolio_backtest(close, sig, cap, N_HOLD, costs=costs, members_asof=asof,
                                         weight_asof=wfn, rebalance_band=band)

    # 2) COMBINE: dense value:reversal sweep at 1M, NET and GROSS Calmar. Light reversal
    #    tilts (value ~80-90%) beat both the value baseline (0.28) and the A2 inverse-vol
    #    win (0.30); GROSS rising confirms it is real alpha. Heavy reversal (50/50) is worse
    #    -- too much of a high-turnover, low-return leg.
    cap = 1_000_000
    rows = []
    print(f"2) value:reversal weight sweep ({cap:,}); NET = A-share frictions, GROSS = zero-cost:")
    print(f"  {'variant':>22} {'CAGR':>8} {'maxDD':>8} {'netCal':>7} {'grossCal':>8} {'costs':>10}")

    def line(tag, sig, wfn=None):
        rn, rg = bt(sig, cap, None, wfn), bt(sig, cap, ZERO_COSTS, wfn)
        print(f"  {tag:>22} {rn.cagr:>+8.1%} {rn.max_drawdown:>8.1%} {cal(rn):>7.2f} {cal(rg):>8.2f} "
              f"{rn.total_costs:>10,.0f}")
        rows.append({"variant": tag, "cagr": rn.cagr, "max_dd": rn.max_drawdown,
                     "net_calmar": cal(rn), "gross_calmar": cal(rg), "total_costs": rn.total_costs})

    line("value (base)", ep)
    line("value/invvol (A2)", ep, invv)
    for wv in [1, 2, 3, 4, 5, 7, 9]:
        line(f"val+rev1 {wv}/1", fl.blend([ep_pit, rev1_pit], [wv, 1]))
    for wv in [2, 3, 4]:                      # shorter reversal: directionally stronger
        line(f"val+rev10 {wv}/1", fl.blend([ep_pit, rev10_pit], [wv, 1]))
    print()

    # 3) ROBUSTNESS across capital tiers (the plateau, not a single lucky point).
    print("3) cross-tier NET Calmar (robustness of the tilt, not a single point):")
    print(f"  {'tier':>12} {'value':>7} {'rev1 5/1':>9} {'rev1 7/1':>9} {'rev10 3/1':>10}")
    for c in TIERS:
        v, a, b, d = (bt(ep, c), bt(fl.blend([ep_pit, rev1_pit], [5, 1]), c),
                      bt(fl.blend([ep_pit, rev1_pit], [7, 1]), c), bt(fl.blend([ep_pit, rev10_pit], [3, 1]), c))
        print(f"  {c:>12,} {cal(v):>7.2f} {cal(a):>9.2f} {cal(b):>9.2f} {cal(d):>10.2f}")
        for tag, r in [("value", v), ("rev1 5/1", a), ("rev1 7/1", b), ("rev10 3/1", d)]:
            rows.append({"variant": f"tier{c} {tag}", "cagr": r.cagr, "max_dd": r.max_drawdown,
                         "net_calmar": cal(r), "gross_calmar": float("nan"), "total_costs": r.total_costs})
    print()

    # 4) Turnover buffer on the sweet-spot tilt -- same verdict as for value: it HURTS
    #    (value-driven strategies must rotate; hysteresis strands capital in dear names).
    print(f"4) turnover buffer on val+rev1 5/1 ({cap:,}) -- still hurts value-driven rotation:")
    print(f"  {'band':>6} {'CAGR':>8} {'maxDD':>8} {'Calmar':>7} {'costs':>10}")
    sweet = fl.blend([ep_pit, rev1_pit], [5, 1])
    for band in [0, 5, 10, 20]:
        r = bt(sweet, cap, band=band)
        print(f"  {band:>6} {r.cagr:>+8.1%} {r.max_drawdown:>8.1%} {cal(r):>7.2f} {r.total_costs:>10,.0f}")
    pd.DataFrame(rows).to_csv(BACKTESTS_DIR / "multifactor_pit.csv", index=False)

    print("\nFinding: a MODEST reversal tilt (value ~80-90%, e.g. 5/1-7/1) raises net Calmar "
          "0.28 -> ~0.32, ABOVE the A2 inverse-vol win (0.30), across the 4/1-9/1 plateau and "
          "all capital tiers; GROSS Calmar rises too (0.30 -> 0.34), so it is genuine pre-cost "
          "alpha, not luck surviving frictions. A 10-day reversal is stronger (net 0.35) but "
          "picking the exact window/weight in-sample risks overfitting -- take 5/1-7/1 (0.32) as "
          "the robust read. KEY CAVEAT: the gain is all in the NUMERATOR (CAGR ~+1pp); maxDD "
          "stays ~-33%, so the systematic drawdown is NOT cured. Inverse-vol doesn't stack "
          "(reversal already supplies the diversification) and the turnover buffer hurts. To "
          "cut the drawdown itself, untried levers remain: sector-neutralization, the deferred "
          "size factor, and a wider universe (CSI 500/1000) -- all needing new data ingestion.")


if __name__ == "__main__":
    main()
