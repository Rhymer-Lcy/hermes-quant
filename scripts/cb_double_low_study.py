"""The pre-registered CB double-low study -- run EXACTLY as frozen in docs/cb_lake.md.

Order of operations is part of the pre-registration: the two lake cross-checks and the
turnover-floor calibration run and PRINT before any strategy return is computed, so no
knob can turn after seeing performance. Implementation choices the freeze left open are
fixed here, still ahead of results: the rolling turnover median needs >=10 traded days
inside its 20-day window; the floor percentile is taken PER EXCHANGE (Sina's volume unit
is not guaranteed consistent across SH/SZ -- a within-market percentile is unit-invariant
either way); ties in the double-low score break by code order.

Prereq: python scripts/build_cb_lake.py    (the JSL revision sample is pulled here)

    python scripts/cb_double_low_study.py
"""
import numpy as np
import pandas as pd

from hermes.cb import data as cb
from hermes.cb.backtest import double_low_backtest
from hermes.cb.checks import close_mismatch, revision_jump_matched
from hermes.paths import BACKTESTS_DIR

WINDOW_START = "2018-01-01"
STRESS = ("2022-08-01", "2024-12-31")   # rule change + credit events sub-window
N_HOLD = 20
N_HOLD_SMALL = 10
COST_PER_SIDE = 0.0005                  # 0.10% round trip; sensitivity at 0.20%
MIN_HISTORY_DAYS = 60
FLOOR_PERCENTILE = 10
REVISION_SAMPLE = 120
SEED = 20260711
DEFAULTED = frozenset({"128100", "123015"})   # Soute, Landun: the 2023 credit delistings


def month_end_signals(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    ends = pd.Series(calendar, index=calendar).groupby(calendar.to_period("M")).max()
    return pd.DatetimeIndex(ends[ends >= WINDOW_START])


def net(equity: pd.Series, start: str, end: str) -> float:
    part = equity.loc[start:end]
    return float(part.iloc[-1] / part.iloc[0] - 1.0) if len(part) > 1 else float("nan")


def main() -> None:
    bars, prem = cb.load_bars(), cb.load_premium()
    codes = sorted(set(bars["code"]) & set(prem["code"]))
    bars, prem = bars[bars["code"].isin(codes)], prem[prem["code"].isin(codes)]
    print(f"lake: {len(codes)} bonds with both bars and premium series")

    # --- cross-check 1: Sina close vs Eastmoney close (docs/cb_lake.md) ----------------
    cm = close_mismatch(bars, prem)
    print(f"close cross-check: {cm['rows']} joined rows, "
          f"mismatch(>0.5%) rate {cm['mismatch_rate']:.4%}, worst {cm['worst']:.2%}")

    # --- cross-check 2: EM conversion-value jumps vs the JSL revision log --------------
    rng = np.random.default_rng(SEED)
    last_bar = bars.groupby("code")["date"].max()
    in_window = sorted(last_bar[last_bar >= WINDOW_START].index)
    sample = sorted(set(rng.choice(in_window, size=min(REVISION_SAMPLE, len(in_window)),
                                   replace=False)) | (DEFAULTED & set(codes)))
    cb.build_revisions(sample)
    logs = cb.load_revisions()
    cv = prem.pivot(index="date", columns="code", values="conv_value")
    matched = unresolved = 0
    events = 0
    for row in logs.itertuples():
        if row.code not in cv.columns:
            continue
        got = revision_jump_matched(cv[row.code], row.old_conv_price, row.new_conv_price,
                                    row.effective_date)
        events += 1
        if got is None:
            unresolved += 1
        elif got:
            matched += 1
    resolved = events - unresolved
    print(f"revision cross-check: {len(logs)} logged revisions on {logs['code'].nunique()} "
          f"sampled bonds; {resolved} resolved, match rate "
          f"{matched / resolved:.1%}" if resolved else "revision cross-check: none resolved")

    # --- panels and eligibility (floor fixed before any return is computed) ------------
    close = bars.pivot(index="date", columns="code", values="close").sort_index()
    volume = bars.pivot(index="date", columns="code", values="volume").sort_index()
    em_close = prem.pivot(index="date", columns="code", values="close").reindex(close.index)
    em_premium = (prem.pivot(index="date", columns="code", values="conv_premium")
                  .reindex(close.index))
    calendar = close.index
    signals = month_end_signals(calendar)

    history_ok = close.notna().cumsum().shift(1) >= MIN_HISTORY_DAYS
    turnover20 = (close * volume).rolling(20, min_periods=10).median()
    score_all = em_close + em_premium

    base = close.notna() & history_ok & score_all.notna()
    pool = turnover20[base].loc[signals]
    floors = {}
    for tag, prefix in [("SH", "11"), ("SZ", "12")]:
        cols = [c for c in close.columns if c.startswith(prefix)]
        vals = pool[cols].stack().dropna()
        floors[prefix] = float(np.percentile(vals, FLOOR_PERCENTILE))
        print(f"turnover floor {tag}: p{FLOOR_PERCENTILE} = {floors[prefix]:,.0f} "
              f"(median {vals.median():,.0f}, {len(vals)} bond-months)")
    floor_row = pd.Series({c: floors[c[:2]] for c in close.columns})
    eligible = base & turnover20.ge(floor_row, axis=1)

    score = score_all.where(eligible).loc[signals].dropna(how="all")
    n_elig = score.notna().sum(axis=1)
    print(f"eligible per rebalance: min {n_elig.min()}, median {int(n_elig.median())}, "
          f"max {n_elig.max()} over {len(score)} rebalances")

    # --- the frozen runs ----------------------------------------------------------------
    runs = {
        "top20": double_low_backtest(close, score, N_HOLD, COST_PER_SIDE),
        "top20_cost2x": double_low_backtest(close, score, N_HOLD, 2 * COST_PER_SIDE),
        "top20_default0": double_low_backtest(close, score, N_HOLD, COST_PER_SIDE,
                                              zero_mark=DEFAULTED),
        "top10": double_low_backtest(close, score, N_HOLD_SMALL, COST_PER_SIDE),
        "bench_ew": double_low_backtest(close, score, None, 0.0),
    }
    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  {'run':>14} {'net total':>10} {'CAGR':>7} {'maxDD':>7} {'stress net':>10} "
          f"{'turnover':>8}")
    for name, r in runs.items():
        r.equity.to_csv(BACKTESTS_DIR / f"cb_double_low_{name}.csv")
        print(f"  {name:>14} {r.total_return:>+10.1%} {r.cagr:>+7.1%} {r.max_drawdown:>7.1%} "
              f"{net(r.equity, *STRESS):>+10.1%} "
              f"{r.rebalances['oneway_turnover'].mean():>8.1%}")

    # --- the pre-registered verdict (all three must hold; docs/cb_lake.md) --------------
    strat, bench = runs["top20"], runs["bench_ew"]
    crit = {
        "net > EW universe": strat.total_return > bench.total_return,
        "stress net > 0": net(strat.equity, *STRESS) > 0,
        "maxDD within 5pp of universe": strat.max_drawdown >= bench.max_drawdown - 0.05,
    }
    for name, ok in crit.items():
        print(f"  criterion {name}: {'PASS' if ok else 'FAIL'}")
    print(f"VERDICT: double-low {'SURVIVED' if all(crit.values()) else 'DID NOT SURVIVE'} "
          f"(pre-registered criteria, {WINDOW_START} -> {calendar.max().date()})")


if __name__ == "__main__":
    main()
