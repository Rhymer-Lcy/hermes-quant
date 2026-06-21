"""Survivorship bias, isolated: point-in-time vs current HS300 membership.

Same strategy, same union price panel, same backtest code -- only the membership
rule differs:
  - biased: restrict to TODAY's HS300 members for the whole history (look-ahead).
  - PIT:    restrict to the THEN-CURRENT members at each rebalance (no look-ahead).
The gap (biased minus PIT) is the survivorship inflation, made explicit.

Prereq: python scripts/build_pit_dataset.py  (membership + union price history)

    python scripts/survivorship_study.py
"""
import pandas as pd

from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.research import CAPITAL_TIERS
from hermes.research.backtest.portfolio import momentum_portfolio_backtest

N_HOLD = 10
LOOKBACK = 20


def main() -> None:
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    panel = load_close_panel(codes=union)
    print(f"union: {len(union)} names ever in HS300, {mdf['date'].nunique()} monthly "
          f"snapshots, panel {panel.shape[0]} days\n")

    current = set(mdf.loc[mdf["date"] == mdf["date"].max(), "code"])
    asof_pit = membership_lookup(mdf)
    asof_biased = lambda _when: current  # noqa: E731 -- today's members, applied to the past

    print(f"  {'capital':>10} {'mode':>7} {'held':>6} {'net total':>12} {'net CAGR':>9} {'maxDD':>7}")
    for cap in CAPITAL_TIERS:
        for tag, asof in [("biased", asof_biased), ("PIT", asof_pit)]:
            r = momentum_portfolio_backtest(panel, cap, N_HOLD, LOOKBACK, members_asof=asof)
            print(f"  {cap:>10,} {tag:>7} {r.avg_names_held:>6.1f} "
                  f"{r.total_return:>+12.1%} {r.cagr:>+9.1%} {r.max_drawdown:>7.1%}")

    print("\nThe biased rows use TODAY's HS300 (winners) over 2015-2025 = look-ahead in "
          "the universe. PIT uses the then-current members (incl. later removed/delisted), "
          "so it is survivorship-free. biased-minus-PIT CAGR is the inflation -- treat only "
          "the PIT figures as remotely realistic (and still not validated alpha).")


if __name__ == "__main__":
    main()
