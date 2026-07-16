"""The dividend-yield band on banks: buy above 5.5%, sell below 4.5%. Issue #9.

The friend's own if-then, pre-registered before any dividend data was pulled:

  universe   the CSRC-J66 (banking) names present in the PIT HS300+CSI500 lake
  yield      trailing-365-day per-share pre-tax cash dividends (counted from EX-DATE)
             divided by the UNADJUSTED close; strategy returns use the adjusted series
  rule       per name, hysteresis band: enter at yield >= 5.5%, exit below 4.5%
  portfolio  equal weight across held banks, monthly rebalance, signal t -> execute t+1,
             proportional retail costs; cash at 0% when nothing is held
  verdict    CONFIRMED only if P_band beats equal-weight-all-banks net with
             monthly-clustered t > 2

Builds its two small lakes (dividends.parquet, raw_close.parquet) on first run.

    conda activate hermes
    python scripts/dividend_band_study.py
"""
from __future__ import annotations

import pandas as pd

from hermes.data.fundamentals import (DIVIDENDS_PARQUET, INDUSTRY_PARQUET, RAW_CLOSE_PARQUET,
                                      pull_dividends, pull_raw_close, trailing_yield)
from hermes.data.lake import load_close_panel
from hermes.io import atomic_to_parquet
from hermes.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from hermes.research.backtest.metrics import clustered_tstat, sharpe
from hermes.research.backtest.rule_portfolio import monthly_ew_backtest, state_mask

ENTER_YIELD, EXIT_YIELD = 0.055, 0.045
FIRST_EX_YEAR = 2013            # seed the trailing year before the 2015 window opens
EXAMPLE = "sh.600036"           # the friend's literal single-name example
ERAS = [("2015-01-01", "2019-12-31"), ("2020-01-01", "2026-12-31")]


def bank_codes() -> list[str]:
    ind = pd.read_parquet(INDUSTRY_PARQUET)
    banks = set(ind.loc[ind["csrc"] == "J66", "code"])
    lake = {p.stem.replace("_", ".", 1) for p in (PARQUET_DIR / "daily").glob("*.parquet")}
    return sorted(banks & lake)


def ensure_data(codes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if DIVIDENDS_PARQUET.exists() and RAW_CLOSE_PARQUET.exists():
        div = pd.read_parquet(DIVIDENDS_PARQUET)
        raw = pd.read_parquet(RAW_CLOSE_PARQUET)
        if set(codes) <= set(raw.columns):
            return div, raw
    years = range(FIRST_EX_YEAR, pd.Timestamp.now().year + 1)
    print(f"building dividend/raw-close lakes for {len(codes)} banks ...")
    div = pull_dividends(codes, years)
    raw = pull_raw_close(codes)
    return div, raw


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
    print(f"  {'variant':>28} {'wealth':>8} {'bench':>8} {'active/yr':>10} {'Sharpe':>7} "
          f"{'bench':>7} {'t(month)':>9}")
    for r in rows:
        print(f"  {r['tag']:>28} {r['wealth_strat']:>8.3f} {r['wealth_bench']:>8.3f} "
              f"{r['ann_active']:>+10.2%} {r['sharpe_strat']:>7.2f} {r['sharpe_bench']:>7.2f} "
              f"{r['t_month']:>9.2f}")


def run_band(yld: pd.DataFrame, ret: pd.DataFrame, base: pd.DataFrame,
             enter: float, exit_: float) -> pd.Series:
    held = state_mask(yld >= enter, yld < exit_) & base
    return monthly_ew_backtest(held, ret)["net"]


def main() -> None:
    ensure_dirs()
    codes = bank_codes()
    div, raw = ensure_data(codes)
    print(f"banks: {len(codes)}; dividend events: {len(div)}; "
          f"raw closes {raw.index.min().date()} -> {raw.index.max().date()}")

    close = load_close_panel(codes=codes, field="close")
    st = load_close_panel(codes=codes, field="isST")
    ret = close.pct_change(fill_method=None)
    base = ~st.eq(True) & close.notna() & (close.notna().cumsum().shift(1) >= 20)
    yld = trailing_yield(div, raw).reindex(close.index)

    held = state_mask(yld >= ENTER_YIELD, yld < EXIT_YIELD) & base
    band = monthly_ew_backtest(held, ret)["net"]
    bh = monthly_ew_backtest(base, ret)["net"]
    print(f"held banks/day: {held.sum(axis=1).mean():.1f} of {base.sum(axis=1).mean():.1f}; "
          f"days fully in cash: {(held.sum(axis=1) == 0).mean():.1%}")

    rows = [_line("P_band vs all banks NET", band, bh)]
    for s, e in ERAS:
        sl = slice(pd.Timestamp(s), pd.Timestamp(e))
        rows.append(_line(f"{s[:4]}-{e[:4]}", band.loc[sl], bh.loc[sl]))
    for en, ex in [(0.05, 0.04), (0.05, 0.05), (0.06, 0.04), (0.06, 0.05)]:
        rows.append(_line(f"thresholds {en:.1%}/{ex:.1%}",
                          run_band(yld, ret, base, en, ex), bh))
    cmb_band = run_band(yld[[EXAMPLE]], ret[[EXAMPLE]], base[[EXAMPLE]],
                        ENTER_YIELD, EXIT_YIELD)
    cmb_bh = monthly_ew_backtest(base[[EXAMPLE]], ret[[EXAMPLE]])["net"]
    rows.append(_line("the single-name example", cmb_band, cmb_bh))
    _print(rows, "DIVIDEND-YIELD BAND ON BANKS (enter >= 5.5%, exit < 4.5%):")

    head = rows[0]
    verdict = head["ann_active"] > 0 and head["t_month"] > 2
    print(f"\nVERDICT: {'CONFIRMED' if verdict else 'REJECTED'} -- needs P_band > all-banks "
          f"EW net with monthly-clustered t > 2 (got {head['t_month']:.2f})")

    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "dividend_band_summary.parquet",
                      index=False)
    atomic_to_parquet(pd.DataFrame({"band_net": band, "all_banks": bh}),
                      BACKTESTS_DIR / "dividend_band_daily.parquet")
    print(f"saved -> {BACKTESTS_DIR / 'dividend_band_summary.parquet'}")


if __name__ == "__main__":
    main()
