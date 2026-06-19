"""First friction demo: double-MA on one HS300 name across capital tiers.

Shows GROSS (frictionless) vs NET (A-share costs) so the small-account cost drag
is explicit. MECHANISM demo only — a single-name MA crossover has no real edge.

    python scripts/backtest_demo.py
"""
import pandas as pd

from hermes.paths import PARQUET_DIR
from hermes.research import CAPITAL_TIERS
from hermes.research.backtest.frictions import AShareCosts, ZERO_COSTS
from hermes.research.backtest.single_name import double_ma_backtest

CODE = "sh.600000"
TIERS = CAPITAL_TIERS


def main() -> None:
    fp = PARQUET_DIR / "daily" / f"{CODE.replace('.', '_')}.parquet"
    prices = pd.read_parquet(fp)
    print(f"{CODE}: {len(prices)} bars, "
          f"{prices['date'].min().date()} .. {prices['date'].max().date()}")

    gross = double_ma_backtest(prices, capital=100_000, costs=ZERO_COSTS)
    print(f"\nGROSS (frictionless) double-MA 20/60: total {gross.total_return:+.1%}, "
          f"CAGR {gross.cagr:+.1%}, maxDD {gross.max_drawdown:.1%}, trades {gross.n_trades}")

    print("\nNET, by starting capital (same strategy, A-share frictions):")
    print(f"  {'capital':>10} {'net total':>11} {'net CAGR':>9} "
          f"{'costs(RMB)':>11} {'cost/cap':>9}")
    for cap in TIERS:
        r = double_ma_backtest(prices, capital=cap, costs=AShareCosts())
        print(f"  {cap:>10,} {r.total_return:>+11.1%} {r.cagr:>+9.1%} "
              f"{r.total_costs:>11,.0f} {r.total_costs / cap:>8.1%}")

    print("\nNote: single-name lot granularity is mild here; the diversification-killing "
          "100-share lot problem shows up in a MULTI-name portfolio (next step). The "
          "visible drag is the 5-RMB minimum commission + stamp tax + slippage, "
          "proportionally heaviest at the smallest accounts (cost/cap 9.8% at 5k vs "
          "5.6% at 500k -- the same strategy, ~75% more cost drag on the small account).")


if __name__ == "__main__":
    main()
