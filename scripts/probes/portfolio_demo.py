"""Multi-name portfolio demo: monthly top-N cross-sectional momentum on HS300,
across capital tiers, with A-share frictions.

Punchline: `held` (effective names held) collapses far below the target N at small
capital, because a 100-share lot of an HS300 name costs thousands of RMB while the
per-name budget at 5k/10 is only a few hundred. Diversification is structurally
impossible at the smallest accounts.

    python scripts/portfolio_demo.py

This is the NON-PIT baseline (current-membership universe). See portfolio_pit_demo.py
for the canonical survivorship-free biased-vs-PIT study.
"""

from hermes.data.lake import load_close_panel
from hermes.research import CAPITAL_TIERS
from hermes.research.backtest.portfolio import momentum_portfolio_backtest

TIERS = CAPITAL_TIERS
N_HOLD = 10
LOOKBACK = 20


def main() -> None:
    panel = load_close_panel()
    print(f"panel: {panel.shape[1]} names, {panel.shape[0]} days, "
          f"{panel.index.min().date()} .. {panel.index.max().date()}")
    print(f"strategy: monthly equal-weight top-{N_HOLD} by {LOOKBACK}-day return\n")

    print(f"  {'capital':>10} {'target N':>8} {'held(avg)':>10} {'net total':>11} "
          f"{'net CAGR':>9} {'maxDD':>7} {'costs':>10} {'cost/cap':>9}")
    for cap in TIERS:
        r = momentum_portfolio_backtest(panel, capital=cap, n_hold=N_HOLD, lookback=LOOKBACK)
        print(f"  {cap:>10,} {r.target_n_hold:>8} {r.avg_names_held:>10.1f} "
              f"{r.total_return:>+11.1%} {r.cagr:>+9.1%} {r.max_drawdown:>7.1%} "
              f"{r.total_costs:>10,.0f} {r.total_costs / cap:>8.1%}")

    print("\n[WARNING] Net returns are INFLATED by survivorship bias (CURRENT HS300 "
          "membership applied to 2015-2025). Do NOT read the CAGR as real alpha.")
    print("Bias-robust conclusion (same universe across tiers): 'held(avg)' falls "
          "9.9 -> 2.8 from 3M down to 10k -- a 10k account cannot hold 10 names "
          "(100-share lots) and is forced into concentration; ~50k+ is needed for real "
          "diversification.")
    print("(cost/cap is ~40-50% across ALL tiers here, dominated by this strategy's high "
          "monthly turnover, NOT by account size -- the small-account min-commission "
          "penalty is the single-name demo's lesson, not this one.)")
    print("Next data milestone: point-in-time membership + delisted names to kill the bias.")


if __name__ == "__main__":
    main()
