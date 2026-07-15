"""Margin-balance sentiment timing: buy the trough reversal, sell the peak. Issue #6.

A friend's rule, pre-registered before this script existed: hold the market from the reversal
off a margin-financing-balance trough, sell out when the balance gets high. Frozen design:

  series     SSE daily margin financing balance (SSE-only is the frozen market proxy)
  state      TROUGH-ARMED at <= trailing 250d 30th percentile; BUY on the first day, on or
             after arming, whose 20-day balance change turns positive (arming resets after a
             buy); SELL at >= trailing 250d 80th percentile; otherwise persist; start flat
  timing     balance for day t is published overnight -> signals execute at the t+1 close,
             returns accrue from t+2
  vehicle    equal-weight PIT HS300+CSI500 non-ST portfolio, fully in or out, cash at 0%,
             proportional retail costs on each round trip
  verdict    CONFIRMED only if timing beats buy-and-hold on BOTH final net wealth AND Sharpe,
             with monthly-clustered t > 2 on the daily active difference

    conda activate hermes
    python scripts/margin_timing_study.py
"""
from __future__ import annotations

import pandas as pd

from hermes.data.fundamentals import MARGIN_SSE_PARQUET
from hermes.data.lake import load_close_panel
from hermes.data.membership import (CSI500_MEMBERSHIP_PARQUET, MEMBERSHIP_PARQUET,
                                    membership_lookup)
from hermes.io import atomic_to_parquet
from hermes.paths import BACKTESTS_DIR, ensure_dirs
from hermes.research.backtest.metrics import clustered_tstat, sharpe
from hermes.research.backtest.rule_portfolio import (proportional_rates,
                                                     threshold_reversal_state)

ARM_PCT, SELL_PCT = 0.30, 0.80
LOOKBACK, REVERSAL = 250, 20
ERAS = [("2015-01-01", "2019-12-31"), ("2020-01-01", "2026-12-31")]


def timing_state(balance: pd.Series, arm_pct: float = ARM_PCT,
                 sell_pct: float = SELL_PCT) -> pd.Series:
    """The frozen state machine on the margin-balance series: 1 = invested, 0 = flat."""
    return threshold_reversal_state(balance, arm_pct, sell_pct, LOOKBACK, REVERSAL)


def universe_return() -> pd.Series:
    """Daily equal-weight return of the PIT HS300+CSI500 non-ST universe (the repo's standard
    benchmark construction, as in the limit-up study)."""
    hs = pd.read_parquet(MEMBERSHIP_PARQUET)
    cs = pd.read_parquet(CSI500_MEMBERSHIP_PARQUET)
    union = sorted(set(hs["code"]) | set(cs["code"]))
    close = load_close_panel(codes=union, field="close")
    st = load_close_panel(codes=union, field="isST")
    hs_asof, cs_asof = membership_lookup(hs), membership_lookup(cs)
    member = pd.DataFrame({c: [c in hs_asof(d) or c in cs_asof(d) for d in close.index]
                           for c in close.columns}, index=close.index)
    universe = member & ~st.eq(True) & close.notna() & (close.notna().cumsum().shift(1) >= 20)
    ret = close.pct_change(fill_method=None)
    return ret.where(universe).mean(axis=1).dropna()


def run_arm(state: pd.Series, uni: pd.Series, buy_rate: float, sell_rate: float) -> pd.Series:
    """Net daily strategy return for a 0/1 state series on the margin-date axis: positions
    take effect two trading days after the balance date (publish overnight, trade t+1 close,
    earn from t+2); trades pay the proportional rates on the whole portfolio."""
    st_daily = state.reindex(uni.index, method="ffill").fillna(0.0)
    pos = st_daily.shift(2).fillna(0.0)
    trade = st_daily.shift(1).fillna(0.0).diff().fillna(0.0)
    cost = trade.clip(lower=0.0) * buy_rate + (-trade.clip(upper=0.0)) * sell_rate
    return pos * uni - cost


def _line(tag: str, strat: pd.Series, bh: pd.Series) -> dict:
    active = strat - bh
    return {"tag": tag,
            "wealth_strat": float((1 + strat).prod()),
            "wealth_bh": float((1 + bh).prod()),
            "sharpe_strat": sharpe(strat), "sharpe_bh": sharpe(bh),
            "t_month": clustered_tstat(active, active.index, freq="M"),
            "days_in": float((strat != 0).mean())}


def _print(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'variant':>28} {'wealth':>8} {'vs b&h':>8} {'Sharpe':>7} {'vs b&h':>7} "
          f"{'t(month)':>9} {'in-mkt':>7}")
    for r in rows:
        print(f"  {r['tag']:>28} {r['wealth_strat']:>8.3f} {r['wealth_bh']:>8.3f} "
              f"{r['sharpe_strat']:>7.2f} {r['sharpe_bh']:>7.2f} {r['t_month']:>9.2f} "
              f"{r['days_in']:>7.1%}")


def main() -> None:
    ensure_dirs()
    margin = pd.read_parquet(MARGIN_SSE_PARQUET).set_index("date")["rzye"]
    uni = universe_return()
    buy_rate, sell_rate = proportional_rates()
    print(f"SSE margin balance: {len(margin)} days, "
          f"{margin.index.min().date()} -> {margin.index.max().date()}")

    state = timing_state(margin)
    strat = run_arm(state, uni, buy_rate, sell_rate)
    bh = uni.copy()
    flips = state.diff().abs().sum()
    print(f"round trips: {int(flips // 2)} (position changes: {int(flips)})")

    rows = [_line("frozen rule NET", strat, bh)]

    # Frozen attribution legs (reported, not the verdict).
    first_buy = state.eq(1).idxmax() if state.eq(1).any() else None
    entry_only = pd.Series(0.0, index=state.index)
    if first_buy is not None:
        entry_only.loc[first_buy:] = 1.0
    hi = margin.rolling(LOOKBACK).quantile(SELL_PCT)
    exit_only = (~(margin >= hi)).astype(float)
    rows.append(_line("entry leg only", run_arm(entry_only, uni, buy_rate, sell_rate), bh))
    rows.append(_line("exit leg only", run_arm(exit_only, uni, buy_rate, sell_rate), bh))

    for s, e in ERAS:
        sl = slice(pd.Timestamp(s), pd.Timestamp(e))
        rows.append(_line(f"{s[:4]}-{e[:4]}", strat.loc[sl], bh.loc[sl]))

    # Frozen sensitivity read: +-10 percentile points on both thresholds (fragility, not model
    # selection -- the verdict uses the frozen 30/80 pair only).
    for a, sp in [(0.20, 0.70), (0.20, 0.90), (0.40, 0.70), (0.40, 0.90)]:
        alt = run_arm(timing_state(margin, arm_pct=a, sell_pct=sp), uni, buy_rate, sell_rate)
        rows.append(_line(f"thresholds {int(a*100)}/{int(sp*100)}", alt, bh))

    _print(rows, "MARGIN-BALANCE TIMING vs buy-and-hold (EW PIT universe):")

    head = rows[0]
    verdict = (head["wealth_strat"] > head["wealth_bh"]
               and head["sharpe_strat"] > head["sharpe_bh"] and head["t_month"] > 2)
    print(f"\nVERDICT (frozen rule): {'CONFIRMED' if verdict else 'REJECTED'} -- needs wealth "
          f"AND Sharpe above buy-and-hold with monthly-clustered t > 2")

    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "margin_timing_summary.parquet",
                      index=False)
    atomic_to_parquet(pd.DataFrame({"strat_net": strat, "buy_hold": bh}),
                      BACKTESTS_DIR / "margin_timing_daily.parquet")
    print(f"saved -> {BACKTESTS_DIR / 'margin_timing_summary.parquet'}")


if __name__ == "__main__":
    main()
