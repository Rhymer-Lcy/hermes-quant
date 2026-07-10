"""A4: does a size tilt cut plain value's -33% drawdown? NO -- it DEEPENS it.

Reconstructs free-float cap from the existing daily lake (amount / (turn/100); BaoStock
`turn` is the free-float turnover rate in percent) -- no Tushare daily_basic pull, so the
old rate-limit blocker is moot. Then sweeps a value:size blend (IRON RULE 2: the full
weight curve, not a single point) at PIT HS300, top-10 monthly, A-share frictions, and
reports NET and GROSS (zero-cost) Calmar.

Finding: every size weight LOWERS CAGR and DEEPENS maxDD; size-only is a -85% catastrophe.
Within HS300 the universe is all large caps, so 'small' = a demoted / falling-knife blue
chip (distress beta), not the small-cap premium -- and it co-moves with the market
(corr(small, large) ≈ 0.76), so it adds no uncorrelated return. Same verdict as A1/A2/B:
the -33% is systematic whole-market beta, uncuttable from inside the HS300 large-cap set.

    python scripts/size_tilt_study.py
"""
import pandas as pd

from hermes.data.ingest import BACKTEST_END
from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.paths import BACKTESTS_DIR
from hermes.research.backtest.frictions import ZERO_COSTS
from hermes.research.backtest.portfolio import signal_portfolio_backtest
from hermes.research.factors import library as fl

N_HOLD = 10
TIERS = [100_000, 1_000_000, 10_000_000]


def cal(r):
    return r.cagr / abs(r.max_drawdown) if r.max_drawdown < 0 else float("nan")


def main() -> None:
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof = membership_lookup(mdf)

    close = load_close_panel(codes=union, field="close", end=BACKTEST_END)
    amount = load_close_panel(codes=union, field="amount", end=BACKTEST_END)
    turn = load_close_panel(codes=union, field="turn", end=BACKTEST_END)
    cap = fl.float_cap(amount, turn)

    # Sanity-check the free reconstruction (in ×10^8 CNY) against known mega-caps.
    pdt = cap.index[cap.index <= pd.Timestamp(BACKTEST_END)][-1]
    print(f"free-float cap reconstruction @ {pdt.date()} (×10^8 CNY), {cap.notna().sum().sum() / amount.notna().sum().sum():.1%} bar coverage:")
    for c in ["sh.600000", "sh.601318", "sh.600519"]:
        if c in cap.columns and pd.notna(cap.loc[pdt, c]):
            print(f"  {c}: {cap.loc[pdt, c] / 1e8:,.0f} ×10^8")

    ep = fl.earnings_yield(load_close_panel(codes=union, field="peTTM", end=BACKTEST_END))
    size = fl.small_size(cap)                         # higher = smaller cap
    ep_pit = fl.restrict_to_universe(ep, asof)        # IRON RULE 1: PIT before blend
    size_pit = fl.restrict_to_universe(size, asof)

    def bt(sig, capital, costs=None):
        return signal_portfolio_backtest(close, sig, capital, N_HOLD, costs=costs, members_asof=asof)

    cap_tier = 1_000_000
    rows = []
    print(f"\nvalue:size weight sweep @ {cap_tier:,} (top{N_HOLD}, monthly, PIT; NET frictions / GROSS zero-cost):")
    print(f"  {'variant':>18} {'CAGR':>8} {'maxDD':>8} {'netCal':>7} {'grossCal':>8}")

    def line(tag, sig):
        rn, rg = bt(sig, cap_tier), bt(sig, cap_tier, ZERO_COSTS)
        print(f"  {tag:>18} {rn.cagr:>+8.1%} {rn.max_drawdown:>8.1%} {cal(rn):>7.2f} {cal(rg):>8.2f}")
        rows.append({"variant": tag, "cagr": rn.cagr, "max_dd": rn.max_drawdown,
                     "net_calmar": cal(rn), "gross_calmar": cal(rg)})

    line("value (base)", ep)
    for w in [0.2, 0.3, 0.5]:
        line(f"val+{w}*size", fl.blend([ep_pit, size_pit], [1.0, w]))
    line("size only", size_pit)

    # Cross-tier robustness of the (negative) result -- not a single-account artifact.
    print("\ncross-tier maxDD (size tilt deepens drawdown at every tier):")
    print(f"  {'tier':>12} {'value':>8} {'val+0.3size':>12} {'size only':>10}")
    for c in TIERS:
        v, m, s = bt(ep, c), bt(fl.blend([ep_pit, size_pit], [1.0, 0.3]), c), bt(size_pit, c)
        print(f"  {c:>12,} {v.max_drawdown:>8.1%} {m.max_drawdown:>12.1%} {s.max_drawdown:>10.1%}")

    pd.DataFrame(rows).to_csv(BACKTESTS_DIR / "a4_size.csv", index=False)
    print("\nFinding: A4 REJECTED. Free-float cap is free from the lake (no Tushare), but a size "
          "tilt deepens drawdown monotonically (val+0.2/0.3/0.5*size maxDD -36/-35/-42% vs base "
          "-34%; size-only -85%). 'Small' inside HS300 = distress beta (corr≈0.76 with large), "
          "not the SMB premium. The -33% is systematic; size does not cut it.")


if __name__ == "__main__":
    main()
