"""Does a turnover-stability tilt blend into the deployed HS300 signal the way reversal did?

MOTIVATION. liquidity_factor_study measured low_turnover_vol (turnstd20) at IC t=+4.26 on PIT
HS300 -- the strongest single-factor IC recorded on the deployed universe -- yet its top-10 basket
alone does not beat the deployed book (Calmar 0.29 vs 0.32). That is exactly where the reversal
tilt started (docs/multi_factor.md: a weak-alone leg that ADDED breadth to value, 0.28 -> 0.32).
This asks the same question for turnstd, under the same discipline.

PRE-REGISTERED DESIGN (fixed before running; no post-hoc grid growth):
  1. Diagnosis: monthly rank-IC correlation of turnstd vs the deployed legs (ep, rev1) and
     low_vol. The A2/B lesson: the value family and low_vol are mutually 0.7-0.9 correlated
     (redundant); reversal was the lone negative. The matrix is a SCREEN only -- the combination
     curve below is the evidence (multi_factor's own sampling lesson).
  2. Sweep: blend([ep, rev1, turnstd], [5, 1, w]) for w in {0.5, 1, 2, 3, 5} -- the deployed 5:1
     core held fixed, turnstd added at increasing weight. One substitution probe,
     blend([ep, turnstd], [5, 1]) (does turnstd replace reversal?), as a single pre-registered
     point, not a sweep. turnstd window fixed at 20 (as measured); no window sweep.
  3. Metrics: NET (A-share frictions) and GROSS (zero-cost) Calmar at CNY 1,000,000, maxDD,
     costs. Baselines: value alone; deployed 5:1.
  4. ADOPTION BAR (all must hold, per multi_factor.md): a winner must beat the deployed book on
     net Calmar AND on gross Calmar (real alpha, not friction luck) AND hold across capital tiers
     (100k / 1M / 10M). Anything less is recorded and rejected.
The deployed book is untouched regardless: a passing configuration would become a v2 candidate
with its own forward paper record, never a retroactive edit (the running record stays clean).

    python scripts/turnstd_blend_study.py
"""
from __future__ import annotations

import pandas as pd

from hermes.data.ingest import BACKTEST_END
from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.paths import BACKTESTS_DIR
from hermes.research.backtest.frictions import ZERO_COSTS
from hermes.research.backtest.portfolio import signal_portfolio_backtest
from hermes.research.eval.factor_eval import compute_ic
from hermes.research.factors import library as fl

TIERS = [100_000, 1_000_000, 10_000_000]
N_HOLD = 10
CAP = 1_000_000
WEIGHTS = [0.5, 1.0, 2.0, 3.0, 5.0]


def cal(r) -> float:
    return r.cagr / abs(r.max_drawdown) if r.max_drawdown < 0 else float("nan")


def main() -> None:
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof = membership_lookup(mdf)
    close = load_close_panel(codes=union, field="close", end=BACKTEST_END)
    pe = load_close_panel(codes=union, field="peTTM", end=BACKTEST_END)
    turn = load_close_panel(codes=union, field="turn", end=BACKTEST_END)
    eval_dates = [pd.Timestamp(d) for d in sorted(mdf["date"].unique()) if pd.Timestamp(d) in close.index]

    ep = fl.restrict_to_universe(fl.earnings_yield(pe), asof)
    rev1 = fl.restrict_to_universe(-fl.trailing_return(close, 20), asof)
    turnstd = fl.restrict_to_universe(fl.low_turnover_vol(turn, 20), asof)
    lowvol = fl.restrict_to_universe(fl.low_vol(close, 120), asof)

    # 1) DIAGNOSIS: is turnstd a diversifier (like reversal) or redundant (like low_vol)?
    ics = {k: compute_ic(f, close, eval_dates, members_asof=asof).ic
           for k, f in {"ep": ep, "rev1": rev1, "turnstd": turnstd, "lowvol": lowvol}.items()}
    print("1) monthly rank-IC correlation (screen only; the sweep below is the evidence):")
    print(pd.DataFrame(ics).corr().round(2).to_string())
    print()

    # 2) SWEEP: deployed core [5,1] fixed, turnstd added at increasing weight; net AND gross.
    def bt(sig, capital, costs=None):
        return signal_portfolio_backtest(close, sig, capital, N_HOLD, costs=costs, members_asof=asof)

    rows = []
    print(f"2) blend sweep @ {CAP:,}; NET = A-share frictions, GROSS = zero-cost:")
    print(f"  {'variant':>26} {'CAGR':>8} {'maxDD':>8} {'netCal':>7} {'grossCal':>9} {'costs':>10}")

    def line(tag, sig):
        rn, rg = bt(sig, CAP), bt(sig, CAP, ZERO_COSTS)
        print(f"  {tag:>26} {rn.cagr:>+8.1%} {rn.max_drawdown:>8.1%} {cal(rn):>7.2f} "
              f"{cal(rg):>9.2f} {rn.total_costs:>10,.0f}")
        rows.append({"variant": tag, "cagr": rn.cagr, "max_dd": rn.max_drawdown,
                     "net_calmar": cal(rn), "gross_calmar": cal(rg), "costs": rn.total_costs})
        return sig

    line("value (base)", ep)
    deployed = line("val+rev 5/1 (deployed)", fl.blend([ep, rev1], [5, 1]))
    for w in WEIGHTS:
        line(f"val+rev+tstd 5/1/{w:g}", fl.blend([ep, rev1, turnstd], [5, 1, w]))
    line("val+tstd 5/1 (substitute)", fl.blend([ep, turnstd], [5, 1]))
    print()

    # 3) ROBUSTNESS: any variant beating the deployed NET+GROSS runs the cross-tier gate.
    dep_net = next(r for r in rows if "deployed" in r["variant"])
    winners = [r for r in rows if r["variant"] not in ("value (base)", "val+rev 5/1 (deployed)")
               and r["net_calmar"] > dep_net["net_calmar"]
               and r["gross_calmar"] > dep_net["gross_calmar"]]
    if winners:
        print("3) cross-tier gate for configurations beating the deployed book net AND gross:")
        print(f"  {'variant':>26} {'tier':>12} {'netCal':>7}")
        for r in winners:
            tag = r["variant"]
            if tag.startswith("val+rev+tstd"):
                w = float(tag.split("/")[-1])
                sig = fl.blend([ep, rev1, turnstd], [5, 1, w])
            else:
                sig = fl.blend([ep, turnstd], [5, 1])
            for tier in TIERS:
                rt = bt(sig, tier)
                print(f"  {tag:>26} {tier:>12,} {cal(rt):>7.2f}")
                rows.append({"variant": f"tier{tier} {tag}", "net_calmar": cal(rt)})
    else:
        print("3) no configuration beat the deployed book on net AND gross Calmar; "
              "the cross-tier gate is moot.")

    pd.DataFrame(rows).to_csv(BACKTESTS_DIR / "turnstd_blend.csv", index=False)
    print(f"\nwrote turnstd_blend.csv to {BACKTESTS_DIR}")
    print("Reading: adoption requires beating the deployed 0.32 on NET and GROSS Calmar and "
          "holding across tiers -- the bar the reversal tilt cleared. A high-IC leg that fails "
          "here is another full-distribution defensive signal (the low_vol pattern), and the "
          "deployed book stays exactly as it is. Any passing configuration becomes a v2 candidate "
          "with its OWN forward record; the running paper ledger is never edited.")


if __name__ == "__main__":
    main()
