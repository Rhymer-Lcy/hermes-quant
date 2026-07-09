"""A9 -- per-name stop-loss / take-profit ablation on the deployed HS300 book.

Question: the deployed book rides its positions between monthly rebalances. Does a per-name price
stop cut the intrinsic -33% value-style drawdown that A1-A8 could not, and does a take-profit add
anything? Both are the retail instinct after a drawdown, so the answer needs to be measured, not
assumed.

Method: the deployed signal (value + light 1-month reversal, 5:1, top-10 equal weight, monthly),
survivorship-free PIT HS300, A-share frictions, evaluated with the engine's opt-in stops overlay
(research.backtest.stops). Exits are checked EVERY bar, before any rebalance, against each holding's
share-weighted cost basis; a breached name is liquidated in full and sits in cash until the next
rebalance. Both trigger modes are swept, and their fill conventions are deliberately pessimistic
(gap-through stops fill below the trigger; take-profits never fill above their limit; when one bar
touches both, the stop is assumed) -- so the overlays cannot be flattered by the model.

Reported net AND gross (zero-cost): a lever that only fails net is a friction story, one that fails
gross too is a genuine timing give-up. The take-profit line is additionally sliced by regime, because
its headline bump is claimed to be a single 2015 event rather than a repeatable effect.

    python scripts/stops_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.live.strategy import DEPLOYED, deployed_signal
from hermes.paths import BACKTESTS_DIR
from hermes.research.backtest.frictions import ZERO_COSTS
from hermes.research.backtest.portfolio import signal_portfolio_backtest
from hermes.research.backtest.stops import StopSpec

CAP = 1_000_000
N_HOLD = DEPLOYED.n_hold
# Pin the window: the lake grows daily under the paper feed, so an unpinned study stops reproducing
# the numbers in docs/ (same convention as multifactor_study.py / csi500_universe_study.py).
END = "2025-12-31"

STOP_LEVELS = [0.10, 0.15, 0.20, 0.25]
TAKE_LEVELS = [0.15, 0.20, 0.30, 0.50]
TRIGGERS = ["close", "intraday"]
REGIMES = [("2015-2018", "2015-01-01", "2018-12-31"),
           ("2019-2021", "2019-01-01", "2021-12-31"),
           ("2022-2025", "2022-01-01", "2025-12-31")]


def cal(r) -> float:
    return r.cagr / abs(r.max_drawdown) if r.max_drawdown < 0 else float("nan")


def slice_calmar(equity: pd.Series, lo: str, hi: str) -> tuple[float, float]:
    """Sub-period CAGR / Calmar, drawdown measured against the CARRY-IN high-water mark (a
    window-local reset would understate a book that entered the regime already underwater)."""
    cummax = equity.cummax()
    m = (equity.index >= pd.Timestamp(lo)) & (equity.index <= pd.Timestamp(hi))
    eq = equity[m]
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0
    dd = float((eq / cummax[m] - 1.0).min())
    return float(cagr), (cagr / abs(dd) if dd < 0 else float("nan"))


def main() -> None:
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof = membership_lookup(mdf)

    close = load_close_panel(codes=union, field="close", end=END)
    pe = load_close_panel(codes=union, field="peTTM", end=END)
    high = load_close_panel(codes=union, field="high", end=END)
    low = load_close_panel(codes=union, field="low", end=END)
    sig = deployed_signal(close, pe, asof)

    def run(spec: StopSpec | None, costs=None, collect: bool = False):
        return signal_portfolio_backtest(close, sig, CAP, N_HOLD, costs=costs, members_asof=asof,
                                         stops=spec, high=high, low=low, collect_trades=collect)

    base_net, base_gross = run(None), run(None, ZERO_COSTS)
    b_net, b_gross = cal(base_net), cal(base_gross)
    print(f"A9 -- per-name stops on the deployed book, PIT HS300, {END} window, @ CNY {CAP:,}")
    print(f"baseline (no stops): CAGR {base_net.cagr:+.1%}  maxDD {base_net.max_drawdown:.1%}  "
          f"netCalmar {b_net:.3f}  grossCalmar {b_gross:.3f}\n")

    rows = []
    print(f"  {'overlay':>26} {'trigger':>9} {'CAGR':>7} {'maxDD':>8} {'netCal':>7} {'dNetCal':>8} "
          f"{'grsCal':>7} {'dGrsCal':>8} {'stopExits':>10}")

    def line(tag, spec, trigger):
        rn, rg = run(spec, None, collect=True), run(spec, ZERO_COSTS)
        # count ONLY overlay-forced exits; ordinary rebalance sells and delisting force-exits
        # carry no "reason" and must not be conflated with them
        n_stop = sum(1 for t in rn.trades if t.get("reason") == "stop")
        c_n, c_g = cal(rn), cal(rg)
        print(f"  {tag:>26} {trigger:>9} {rn.cagr:>+7.1%} {rn.max_drawdown:>8.1%} {c_n:>7.3f} "
              f"{c_n - b_net:>+8.3f} {c_g:>7.3f} {c_g - b_gross:>+8.3f} {n_stop:>10}")
        rows.append({"overlay": tag, "trigger": trigger, "cagr": rn.cagr,
                     "max_dd": rn.max_drawdown, "net_calmar": c_n, "d_net_calmar": c_n - b_net,
                     "gross_calmar": c_g, "d_gross_calmar": c_g - b_gross, "n_stop_exits": n_stop})
        return rn

    for trig in TRIGGERS:
        for sl in STOP_LEVELS:
            line(f"stop-loss -{sl:.0%}", StopSpec(stop_loss=sl, trigger=trig), trig)
    for trig in TRIGGERS:
        for tp in TAKE_LEVELS:
            line(f"take-profit +{tp:.0%}", StopSpec(take_profit=tp, trigger=trig), trig)

    df = pd.DataFrame(rows)
    df.to_csv(BACKTESTS_DIR / "stops_study.csv", index=False)

    # Regime stability of the BEST take-profit: a one-event bump should not repeat across regimes.
    best = df[df["overlay"].str.startswith("take-profit")].sort_values("d_net_calmar").iloc[-1]
    tp = float(best["overlay"].split("+")[1].rstrip("%")) / 100.0
    spec = StopSpec(take_profit=tp, trigger=str(best["trigger"]))
    tp_net = run(spec)
    print(f"\nregime stability of the best take-profit ({best['overlay']}, {best['trigger']}) -- "
          f"is the bump repeatable or one event?")
    print(f"  {'window':<12} {'baseCAGR':>9} {'tpCAGR':>9} {'baseCal':>8} {'tpCal':>8} {'dCal':>8}")
    for label, lo, hi in REGIMES:
        bc, bcal = slice_calmar(base_net.equity, lo, hi)
        tc, tcal = slice_calmar(tp_net.equity, lo, hi)
        print(f"  {label:<12} {bc:>+9.1%} {tc:>+9.1%} {bcal:>8.3f} {tcal:>8.3f} {tcal - bcal:>+8.3f}")

    print(f"\nwrote stops_study.csv to {BACKTESTS_DIR}")
    print("Read: an overlay is worth adopting only if it lifts net Calmar AND gross Calmar (real "
          "timing, not a cost artifact) AND holds across regimes. A stop that only cuts maxDD while "
          "giving back more CAGR has not improved the book.")


if __name__ == "__main__":
    main()
