"""Step A2: POSITION-LEVEL drawdown control on the value factor, after A1 (index
trend-timing) failed. Two independent levers, evaluated as a 2x2:

  selection : value (1/PE)            vs  value x low-vol blend  (defensive tilt)
  weighting : equal weight            vs  inverse-volatility     (calm names sized up)

Everything else is held fixed -- survivorship-free PIT HS300 universe, A-share
frictions, monthly rebalance, top-10, signal at month-end / exec next day. The bar
is the same as A1: raise Calmar = CAGR / |maxDD| versus plain value (the documented
+9.2% CAGR / -33% maxDD baseline), i.e. cut drawdown by more than it cuts return.

SURVIVORSHIP NOTE (a real bug caught here): the value x low-vol BLEND standardizes
cross-sectionally, so its inputs must be restricted to the PIT members FIRST -- else
the survivorship-defined union (names ever in HS300) leaks into the z-scores and
inflates the result (the blend's Calmar fell 0.32 -> 0.28 once corrected). Inverse-vol
needs no such fix: it weights only the already-selected names by their own return std.

    python scripts/position_sizing_study.py
"""
import pandas as pd

from hermes.data.ingest import BACKTEST_END
from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.paths import BACKTESTS_DIR
from hermes.research.backtest.portfolio import signal_portfolio_backtest
from hermes.research.backtest.sizing import inverse_vol_weighter
from hermes.research.factors import library as fl

TIERS = [100_000, 1_000_000]
N_HOLD = 10
VOL_LOOKBACK = 60      # trailing days for inverse-vol sizing
LOW_VOL_WINDOW = 120   # trailing days for the low-vol selection factor (IC-validated a priori)


def main() -> None:
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof_members = membership_lookup(mdf)

    close = load_close_panel(codes=union, field="close", end=BACKTEST_END)
    value = fl.earnings_yield(load_close_panel(codes=union, field="peTTM", end=BACKTEST_END))
    # PIT-correct blend: restrict each input to the then-current members BEFORE the
    # cross-sectional standardization inside blend() (else the union leaks -- see module
    # docstring). Single-factor `value` ranks raw, so it needs no restriction.
    def pit_blend(window):
        return fl.blend([fl.restrict_to_universe(value, asof_members),
                         fl.restrict_to_universe(fl.low_vol(close, window), asof_members)])
    blended = pit_blend(LOW_VOL_WINDOW)
    invvol = inverse_vol_weighter(close, lookback=VOL_LOOKBACK)

    # 2x2: (selection signal) x (intra-basket weighting). value/equal == A1 baseline.
    variants = [
        ("value / equal",       value,   None),
        ("value / invvol",      value,   invvol),
        ("valxlowvol / equal",  blended, None),
        ("valxlowvol / invvol", blended, invvol),
    ]

    def line(cap, tag, r):
        calmar = r.cagr / abs(r.max_drawdown) if r.max_drawdown < 0 else float("nan")
        print(f"  {cap:>10,} {tag:>20} {r.cagr:>+8.1%} {r.max_drawdown:>8.1%} "
              f"{calmar:>7.2f} {r.avg_names_held:>6.1f}")
        return {"capital": cap, "variant": tag, "cagr": r.cagr, "max_dd": r.max_drawdown,
                "calmar": calmar, "avg_names_held": r.avg_names_held}

    print(f"  {'capital':>10} {'variant':>20} {'CAGR':>8} {'maxDD':>8} {'Calmar':>7} {'held':>6}")
    rows = []
    for cap in TIERS:
        for tag, sig, wfn in variants:
            r = signal_portfolio_backtest(close, sig, cap, N_HOLD, members_asof=asof_members,
                                          weight_asof=wfn)
            rows.append(line(cap, tag, r))
        print()
    pd.DataFrame(rows).to_csv(BACKTESTS_DIR / "value_sizing_pit.csv", index=False)

    # Sensitivity (1M tier). Inverse-vol lookback is robust; the low-vol blend WINDOW,
    # once PIT-corrected, is noisy and non-monotone -- the earlier "clean gradient" was the
    # survivorship leak. We report the curve rather than tuning to the best window (that
    # would be in-sample overfitting, the same trap as survivorship bias).
    cap = 1_000_000
    print(f"\n  inverse-vol LOOKBACK sweep (selection=value, {cap:,}):")
    print(f"  {'lookback':>10} {'CAGR':>8} {'maxDD':>8} {'Calmar':>7}")
    for lb in [20, 40, 60, 90, 120]:
        r = signal_portfolio_backtest(close, value, cap, N_HOLD, members_asof=asof_members,
                                      weight_asof=inverse_vol_weighter(close, lookback=lb))
        print(f"  {lb:>10} {r.cagr:>+8.1%} {r.max_drawdown:>8.1%} {r.cagr / abs(r.max_drawdown):>7.2f}")

    print(f"\n  low-vol blend WINDOW sweep, PIT-standardized (equal wt, {cap:,}):")
    print(f"  {'window':>10} {'CAGR':>8} {'maxDD':>8} {'Calmar':>7}")
    for w in [60, 90, 120, 180, 250]:
        r = signal_portfolio_backtest(close, pit_blend(w), cap, N_HOLD, members_asof=asof_members)
        print(f"  {w:>10} {r.cagr:>+8.1%} {r.max_drawdown:>8.1%} {r.cagr / abs(r.max_drawdown):>7.2f}")

    print("\nCorrected finding: of the two levers, only INVERSE-VOL survives -- a small, "
          "robust Calmar lift (0.28 -> 0.30, almost all from CAGR; DD ~unchanged), grounded "
          "in the low-vol anomaly. The value x low-vol BLEND adds nothing once the "
          "survivorship leak in its standardization is removed (Calmar stays 0.28 at the "
          "a-priori 120d window; the window sweep is noisy). The -33% drawdown is systematic "
          "and is NOT cured by position-level levers -- consistent with A1.")


if __name__ == "__main__":
    main()
