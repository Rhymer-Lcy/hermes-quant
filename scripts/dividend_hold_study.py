"""The high-dividend long hold: does it beat the index? Issue #12.

The friend's allocation core, pre-registered before the full-union dividend lake existed:
high-yield names are the long-term book, expected to beat the SSE 50 / CSI 300 at ~8%/yr;
growth positions are only a kicker on top.

  universe   PIT HS300+CSI500 members, non-ST, >= 20 prior traded days, 2015+
  yield      trailing-365d pre-tax cash DPS (counted from EX-DATE) / UNADJUSTED close;
             a name with no trailing dividend carries yield 0
  portfolio  month-end top QUINTILE by trailing yield, EW, execute t+1 close, retail costs
  verdict    CONFIRMED only if the portfolio beats the EW universe net with
             monthly-clustered t > 2 (the CSI300 the friend names is a PRICE index that
             excludes the very dividends this sleeve collects -- reported as a descriptive
             read only, alongside the 8% absolute expectation)

    conda activate hermes
    python scripts/dividend_hold_study.py
"""
from __future__ import annotations

import pandas as pd

from hermes.data.fundamentals import DIVIDENDS_PARQUET, RAW_CLOSE_PARQUET, trailing_yield
from hermes.data.lake import load_close_panel
from hermes.data.membership import (CSI500_MEMBERSHIP_PARQUET, MEMBERSHIP_PARQUET,
                                    membership_lookup)
from hermes.data.sw_indices import HS300, SW_INDICES_PARQUET
from hermes.io import atomic_to_parquet
from hermes.paths import BACKTESTS_DIR, ensure_dirs
from hermes.research.backtest.metrics import clustered_tstat, sharpe
from hermes.research.backtest.rule_portfolio import monthly_ew_backtest

TOP_Q, TOP_D = 0.80, 0.90
EXPECTED_CAGR = 0.08
ERAS = [("2015-01-01", "2019-12-31"), ("2020-01-01", "2026-12-31")]


def _line(tag: str, strat: pd.Series, bench: pd.Series) -> dict:
    active = (strat - bench).dropna()
    return {"tag": tag,
            "wealth_strat": float((1 + strat.dropna()).prod()),
            "wealth_bench": float((1 + bench.dropna()).prod()),
            "ann_active": float(active.mean() * 252),
            "sharpe_strat": sharpe(strat), "sharpe_bench": sharpe(bench),
            "t_month": clustered_tstat(active, active.index, freq="M")}


def _print(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'variant':>30} {'wealth':>8} {'bench':>8} {'active/yr':>10} {'Sharpe':>7} "
          f"{'bench':>7} {'t(month)':>9}")
    for r in rows:
        print(f"  {r['tag']:>30} {r['wealth_strat']:>8.3f} {r['wealth_bench']:>8.3f} "
              f"{r['ann_active']:>+10.2%} {r['sharpe_strat']:>7.2f} {r['sharpe_bench']:>7.2f} "
              f"{r['t_month']:>9.2f}")


def main() -> None:
    ensure_dirs()
    hs = pd.read_parquet(MEMBERSHIP_PARQUET)
    cs = pd.read_parquet(CSI500_MEMBERSHIP_PARQUET)
    union = sorted(set(hs["code"]) | set(cs["code"]))
    close = load_close_panel(codes=union, field="close")
    st = load_close_panel(codes=union, field="isST")
    hs_asof, cs_asof = membership_lookup(hs), membership_lookup(cs)
    member = pd.DataFrame({c: [c in hs_asof(d) or c in cs_asof(d) for d in close.index]
                           for c in close.columns}, index=close.index)
    base = (member & ~st.eq(True) & close.notna()
            & (close.notna().cumsum().shift(1) >= 20))
    ret = close.pct_change(fill_method=None)

    div = pd.read_parquet(DIVIDENDS_PARQUET)
    div["ex_date"] = pd.to_datetime(div["ex_date"])
    raw = pd.read_parquet(RAW_CLOSE_PARQUET)
    yld = (trailing_yield(div, raw)
           .reindex(index=close.index, columns=close.columns).fillna(0.0))
    rank = yld.where(base).rank(axis=1, pct=True)

    port = monthly_ew_backtest(rank >= TOP_Q, ret)["net"]
    bh = monthly_ew_backtest(base, ret)["net"]
    print(f"top-quintile names/day: {(rank >= TOP_Q).sum(axis=1).mean():.0f} of "
          f"{base.sum(axis=1).mean():.0f}; mean yield held: "
          f"{yld.where(rank >= TOP_Q).stack().mean():.2%}")

    rows = [_line("top quintile vs EW universe NET", port, bh)]
    for s, e in ERAS:
        sl = slice(pd.Timestamp(s), pd.Timestamp(e))
        rows.append(_line(f"{s[:4]}-{e[:4]}", port.loc[sl], bh.loc[sl]))
    for th in (0.04, 0.055):
        rows.append(_line(f"yield >= {th:.1%}",
                          monthly_ew_backtest(base & (yld >= th), ret)["net"], bh))
    rows.append(_line("top decile", monthly_ew_backtest(rank >= TOP_D, ret)["net"], bh))
    _print(rows, "HIGH-DIVIDEND LONG HOLD (top trailing-yield quintile, monthly EW):")

    # Descriptive reads only (frozen as such): the absolute 8% expectation, and the price
    # index the friend actually names -- which excludes the dividends this sleeve collects.
    yrs = (port.index[-1] - port.index[0]).days / 365.25
    cagr = float((1 + port.dropna()).prod() ** (1 / yrs) - 1)
    hs300 = (pd.read_parquet(SW_INDICES_PARQUET)[HS300]
             .reindex(port.index).pct_change(fill_method=None))
    idx_cagr = float((1 + hs300.dropna()).prod() ** (1 / yrs) - 1)
    act = (port - hs300).dropna()
    print("\ndescriptive reads (no verdict weight):")
    print(f"  portfolio net CAGR {cagr:+.2%} vs the friend's ~{EXPECTED_CAGR:.0%} expectation")
    print(f"  vs the CSI300 PRICE index: index CAGR {idx_cagr:+.2%}, active "
          f"{float(act.mean()) * 252:+.2%}/yr (t_month "
          f"{clustered_tstat(act, act.index, freq='M'):.2f})")

    head = rows[0]
    verdict = head["ann_active"] > 0 and head["t_month"] > 2
    print(f"\nVERDICT: {'CONFIRMED' if verdict else 'REJECTED'} -- needs top-quintile > EW "
          f"universe net with monthly-clustered t > 2 (got {head['t_month']:.2f})")

    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "dividend_hold_summary.parquet",
                      index=False)
    atomic_to_parquet(pd.DataFrame({"top_quintile_net": port, "ew_universe": bh}),
                      BACKTESTS_DIR / "dividend_hold_daily.parquet")
    print(f"saved -> {BACKTESTS_DIR / 'dividend_hold_summary.parquet'}")


if __name__ == "__main__":
    main()
