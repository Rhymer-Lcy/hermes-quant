"""IF (CSI 300 stock index futures) short-hedge overlay on the deployed HS300 monthly book: how much does
hedging the market beta cut the systematic -33% drawdown, and at what cost to return?

A full hedge turns the long book into a market-NEUTRAL book whose return is the value+reversal
spread OVER HS300 -- trading absolute return + market risk for pure alpha + far lower drawdown.
Swept: hedge_ratio 0..1 x futures annual_cost {0,2,4%} (brackets roll/basis carry -- post-2015 IF
ran a deep discount = real negative carry for a short). Hedge leg = HS300 index return (the
beta removed); integer IF contracts (¥300/pt). 2015-2025. `python scripts/index_hedge_study.py`
"""
import numpy as np
import pandas as pd

from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.data.sources import baostock_source as bss
from hermes.live.strategy import deployed_signal
from hermes.research.backtest.hedge import cagr, hedge_overlay, max_drawdown
from hermes.research.backtest.portfolio import signal_portfolio_backtest

END = "2025-12-31"


def ann_vol(eq: pd.Series) -> float:
    return float(eq.pct_change().std() * np.sqrt(252))


def calmar(eq: pd.Series) -> float:
    dd = max_drawdown(eq)
    return cagr(eq) / abs(dd) if dd < 0 else float("nan")


def main() -> None:
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof = membership_lookup(mdf)
    close = load_close_panel(codes=union, field="close", end=END)
    pe = load_close_panel(codes=union, field="peTTM", end=END)
    sig = deployed_signal(close, pe, asof)
    with bss.session():
        index = bss.index_close("sh.000300", "2015-01-01", END)

    # Long book at ¥5M (contract granularity is acceptable there; see the tier table below).
    book = signal_portfolio_backtest(close, sig, 5_000_000, 10, members_asof=asof).equity

    # --- DIAGNOSTIC: is the -33% market-beta (hedgeable) or value-style (not)? ---
    rr = pd.concat([book.pct_change().rename("b"), index.pct_change().rename("m")], axis=1).dropna()
    beta = rr.cov().loc["b", "m"] / rr["m"].var()
    r2 = rr["b"].corr(rr["m"]) ** 2
    resid_eq = (1.0 + (rr["b"] - beta * rr["m"])).cumprod()         # perfect beta-hedge residual (alpha)
    print(f"DIAGNOSTIC: book beta to HS300 = {beta:.2f}, R^2 = {r2:.2f} (only R^2 of variance is market).")
    print(f"  perfect beta-hedge residual maxDD = {max_drawdown(resid_eq):.1%} vs unhedged book "
          f"{max_drawdown(book):.1%} -- if WORSE, the market beta CUSHIONS the drawdown; the -33% is "
          f"value-STYLE, not hedgeable by an index short.\n")

    print("IF short-hedge on deployed HS300 book @ ¥5M (2015-2025); hedge leg = HS300 index return")
    print(f"  {'hedge':>6} {'cost/yr':>7} {'CAGR':>7} {'maxDD':>7} {'Calmar':>7} {'annVol':>7} {'~contr':>7} {'effRatio':>8}")
    for h in (0.0, 0.25, 0.5, 0.75, 1.0):
        for cost in (0.0, 0.02, 0.04):
            heq, nc, eff = hedge_overlay(book, index, h, annual_cost=cost)
            print(f"  {h:>6.2f} {cost:>7.0%} {cagr(heq):>+7.1%} {max_drawdown(heq):>7.1%} "
                  f"{calmar(heq):>7.2f} {ann_vol(heq):>7.1%} {nc:>7.1f} {eff:>8.0%}")
            if h == 0.0:
                break                                   # unhedged: cost irrelevant
        print()

    print("contract-granularity feasibility (full hedge h=1, cost 2%) -- 1 IF ~1.4M RMB notional:")
    print(f"  {'tier':>12} {'~contracts':>11} {'effRatio':>9}  note")
    for tier in (500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000):
        bk = signal_portfolio_backtest(close, sig, tier, 10, members_asof=asof).equity
        _, nc, eff = hedge_overlay(bk, index, 1.0, annual_cost=0.02)
        note = "CANNOT hedge (0 contracts)" if nc < 0.5 else ("coarse (1-2 contracts)" if nc < 3 else "OK granularity")
        print(f"  {tier:>12,} {nc:>11.1f} {eff:>9.0%}  {note}")

    print("\nFinding: the index hedge does NOT cut the -33% -- it WORSENS it at every ratio (maxDD/Calmar "
          "both degrade as h rises), because the book beta is only 0.66 (R^2~0.49) and its drawdown is "
          "value-STYLE, not market-beta: value's worst stretches partly coincide with the index RISING, so "
          "shorting the index adds losses. The market beta CUSHIONS the value drawdown; removing it (a perfect "
          "beta-hedge residual) is WORSE. To cut value-style drawdown you must hedge the STYLE (long cheap / "
          "short expensive = long-short), not the market. (Aside: hedging needs >=~1.4M RMB for 1 IF contract; "
          "small accounts can't anyway.)")


if __name__ == "__main__":
    main()
