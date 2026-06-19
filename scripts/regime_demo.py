"""Step A1: a market-regime (index trend) filter on the value factor, to cut the -33%
drawdown. Compares earnings-yield (value) WITH vs WITHOUT a 沪深300 200-day MA exposure
filter -- survivorship-free (PIT) and with A-share frictions.

    python scripts/regime_demo.py
"""
import pandas as pd

from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.data.sources import baostock_source as bss
from hermes.paths import BACKTESTS_DIR
from hermes.research.backtest.portfolio import signal_portfolio_backtest
from hermes.research.backtest.regime import exposure_lookup, trend_exposure
from hermes.research.factors import library as fl

TIERS = [100_000, 1_000_000]
N_HOLD = 10
MA_WINDOW = 200
INDEX = "sh.000300"


def main() -> None:
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof_members = membership_lookup(mdf)
    close = load_close_panel(codes=union, field="close")
    value = fl.earnings_yield(load_close_panel(codes=union, field="peTTM"))

    with bss.session():
        idx = bss.index_close(INDEX, "2014-01-01", "2025-12-31")   # +1y for MA warm-up
    exposure = trend_exposure(idx, MA_WINDOW)
    asof_regime = exposure_lookup(exposure)
    print(f"{INDEX} {MA_WINDOW}d-MA filter: risk-on {float((exposure.dropna() > 0).mean()):.0%} "
          f"of days\n")

    print(f"  {'capital':>12} {'variant':>14} {'CAGR':>8} {'maxDD':>8} {'Calmar':>7}")
    rows = []
    for cap in TIERS:
        plain = signal_portfolio_backtest(close, value, cap, N_HOLD, members_asof=asof_members)
        filt = signal_portfolio_backtest(close, value, cap, N_HOLD, members_asof=asof_members,
                                         exposure_asof=asof_regime)
        for tag, r in [("value", plain), ("value+regime", filt)]:
            calmar = r.cagr / abs(r.max_drawdown) if r.max_drawdown < 0 else float("nan")
            print(f"  {cap:>12,} {tag:>14} {r.cagr:>+8.1%} {r.max_drawdown:>8.1%} {calmar:>7.2f}")
            rows.append({"capital": cap, "variant": tag, "cagr": r.cagr,
                         "max_dd": r.max_drawdown, "calmar": calmar})
    pd.DataFrame(rows).to_csv(BACKTESTS_DIR / "value_regime_pit.csv", index=False)

    print("\nThe filter earns its keep only if it RAISES Calmar (= CAGR / |maxDD|): cutting "
          "drawdown more than it cuts return. If it just lowers both proportionally, it is not "
          "helping. Next (A2): volatility targeting + per-name weight caps.")


if __name__ == "__main__":
    main()
