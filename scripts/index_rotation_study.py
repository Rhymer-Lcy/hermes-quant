"""Index momentum rotation (dual momentum) across ETF-replicable A-share indices -- a retail
lightweight candidate, tested with the repo's discipline.

WHY THIS SHAPE. The deployed stock book is monthly value+reversal on HS300; A1 showed that
TREND-TIMING that book fails. This is the other classic timing family: CROSS-SECTIONAL momentum
across broad indices, executed monthly via their ETFs (no stock picking, no stamp tax on ETFs,
fractional-lot-free sizes -- the most retail-operable format there is). The bond index is the
defensive leg (absolute-momentum gate): when no equity index beats it, the book stands aside.

PRE-REGISTERED, NO SWEEP: lookbacks 126d and 252-21d (the two standard choices from the momentum
literature) are BOTH reported; top-1 selection; monthly rebalance at month-end signal, next-day
execution (the engine's T+1 convention); 15 bps round-trip cost per switch (ETF commission ~2 bps
per side, no stamp tax, ~5 bps slippage per side, rounded up). Selecting a better lookback after
seeing the results would be in-sample overfitting; if neither pre-registered variant clears the
benchmarks, the family is rejected -- no tuning pass follows.

DATA CAVEATS, stated up front: BaoStock serves no ETF bars, so signals AND returns use the PRICE
indices -- equity legs therefore understate total return by the dividend yield (~2%/yr for HS300),
while the CSI treasury index (sh.000012) accrues coupons; the comparison is biased TOWARD the bond
leg and AGAINST the equity rotation, so a positive equity-rotation read is conservative, and the
real ETF implementation adds tracking difference on top. 000852/000688 are excluded: their ETFs
only became liquid mid-sample, and a universe that changes composition mid-test contaminates the
read.

    python scripts/index_rotation_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.data.ingest import BACKTEST_END
from hermes.data.sources import baostock_source as bss
from hermes.paths import BACKTESTS_DIR

INDICES = {
    "sh.000300": "HS300",        # -> 510300 ETF
    "sh.000905": "CSI500",       # -> 510500 ETF
    "sz.399006": "ChiNext",      # -> 159915 ETF
}
BOND = "sh.000012"               # CSI treasury index -> e.g. 511010 ETF (defensive leg)
START = "2014-01-01"             # one warm-up year before the 2015 backtest start
EVAL_START = "2015-01-01"        # scored window, aligned with the repo's backtest window
LOOKBACKS = {"mom_6m": (126, 0), "mom_12_1": (252, 21)}   # (window, skip) trading days
COST_RT = 15e-4                  # round-trip cost per switch, conservative for ETFs


def month_end_positions(dates: pd.DatetimeIndex) -> list[int]:
    """Positions of each month's last trading day."""
    s = pd.Series(np.arange(len(dates)), index=dates)
    return s.groupby(dates.to_period("M")).last().tolist()


def rotate(closes: pd.DataFrame, window: int, skip: int) -> tuple[pd.Series, pd.Series]:
    """Dual momentum: at each month-end pick the asset (equity indices + bond) with the highest
    trailing (window, skip) return; hold it until the next month-end. Signal at the month-end
    close, position effective from the NEXT trading day's close (T+1). Returns (daily strategy
    returns net of switch costs, the held asset per day)."""
    mom = closes.shift(skip) / closes.shift(window + skip) - 1.0
    ret = closes.pct_change()
    dates = closes.index
    me = month_end_positions(dates)
    held = pd.Series(index=dates, dtype=object)
    for i, pos in enumerate(me):
        row = mom.iloc[pos].dropna()
        if row.empty:
            continue
        pick = row.idxmax()
        nxt = pos + 2                        # signal at close(pos) -> executed close(pos+1):
        stop = me[i + 1] + 2 if i + 1 < len(me) else len(dates)   # returns accrue from pos+2
        held.iloc[nxt:stop] = pick
    strat = pd.Series(0.0, index=dates)
    for d in range(len(dates)):
        a = held.iloc[d]
        if isinstance(a, str):
            strat.iloc[d] = ret[a].iloc[d]
    switches = (held != held.shift()) & held.notna() & held.shift().notna()
    strat[switches] -= COST_RT
    return strat, held


def perf(daily: pd.Series, label: str) -> dict:
    d = daily.dropna()
    eq = (1.0 + d).cumprod()
    years = max((d.index[-1] - d.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0)
    dd = float((eq / eq.cummax() - 1.0).min())
    sharpe = float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else float("nan")
    return {"book": label, "cagr": cagr, "max_dd": dd,
            "calmar": cagr / abs(dd) if dd < 0 else float("nan"), "sharpe": sharpe}


def main() -> None:
    with bss.session():
        closes = pd.concat(
            [bss.index_close(c, START, BACKTEST_END) for c in [*INDICES, BOND]], axis=1).sort_index()
    closes = closes.ffill()                     # indices publish on every session; ffill guards holidays
    print(f"indices: {list(closes.columns)}  {closes.index.min().date()} .. {closes.index.max().date()}")

    rows = []
    eval_mask = closes.index >= pd.Timestamp(EVAL_START)
    for tag, (w, s) in LOOKBACKS.items():
        strat, held = rotate(closes, w, s)
        rows.append(perf(strat[eval_mask], f"rotation {tag}"))
        share = held[eval_mask].value_counts(normalize=True, dropna=True)
        mix = "  ".join(f"{k}:{v:.0%}" for k, v in share.items())
        n_sw = int(((held != held.shift()) & held.notna() & held.shift().notna())[eval_mask].sum())
        print(f"  {tag}: held mix [{mix}]  switches {n_sw} over the window")
    for c in [*INDICES, BOND]:
        rows.append(perf(closes[c].pct_change()[eval_mask], f"B&H {INDICES.get(c, 'Treasury')}"))
    ew = closes[list(INDICES)].pct_change()[eval_mask].mean(axis=1)
    rows.append(perf(ew, "B&H equity EW"))

    res = pd.DataFrame(rows)
    res.to_csv(BACKTESTS_DIR / "index_rotation.csv", index=False)
    print(f"\n{EVAL_START[:4]}-{BACKTEST_END[:4]}, net of {COST_RT * 1e4:.0f} bps per switch; "
          f"PRICE indices (equity legs understate total return by the dividend yield):")
    print(f"  {'book':>22} {'CAGR':>8} {'maxDD':>8} {'Calmar':>7} {'Sharpe':>7}")
    for r in rows:
        print(f"  {r['book']:>22} {r['cagr']:>+8.1%} {r['max_dd']:>8.1%} {r['calmar']:>7.2f} "
              f"{r['sharpe']:>7.2f}")

    print("\nReading: the rotation earns its keep only if it beats BOTH the best single B&H index "
          "AND the equity EW basket on Calmar (risk-adjusted), net of switch costs. A-share index "
          "momentum is regime-whipsaw-prone (cf. A1); if neither pre-registered lookback clears the "
          "bar, the family is REJECTED without a tuning pass.")


if __name__ == "__main__":
    main()
