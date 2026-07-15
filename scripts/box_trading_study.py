"""The 120-day box: trading half the position against a held base. Issue #5.

The friend's rule for the non-cyclical core holdings, pre-registered before this script
existed: draw a box over the prior 120 trading-day closes; lighten to 50% when the close sits
in the top decile of the box, refill to 100% in the bottom decile; between the bands the
position sits still (and its fraction drifts with the price -- no free rebalancing). Signal at
t close, execute at t+1 close, full proportional retail costs including sell-side stamp tax.

  names      the issue #2 quality set (three published annual reports with ROE > 15%),
             restricted to the friend's NON-cyclical buckets; robustness leg: all names ever
             in the PIT HS300 membership
  benchmark  buy-and-hold 100% of the same name over the same window
  verdict    CONFIRMED only if the net daily excess over buy-and-hold is positive with
             monthly-clustered t > 2 on the primary name set; gross-positive-but-net-negative
             is recorded as "the signal exists and the stamp tax owns it"

    conda activate hermes
    python scripts/box_trading_study.py
"""
from __future__ import annotations

import pandas as pd

from hermes.data.fundamentals import CYCLICAL_BUCKETS, INDUSTRY_PARQUET, quality_mask
from hermes.data.lake import load_close_panel
from hermes.data.membership import CSI500_MEMBERSHIP_PARQUET, MEMBERSHIP_PARQUET
from hermes.io import atomic_to_parquet
from hermes.paths import BACKTESTS_DIR, ensure_dirs
from hermes.research.backtest.frictions import ZERO_COSTS, AShareCosts
from hermes.research.backtest.metrics import clustered_tstat, sharpe
from hermes.research.backtest.rule_portfolio import box_target, fractional_target_backtest

BOX_WINDOW = 120
ERAS = [("2015-01-01", "2019-12-31"), ("2020-01-01", "2026-12-31")]


def excess_series(close: pd.DataFrame, costs) -> tuple[pd.Series, pd.DataFrame]:
    """EW daily (box minus hold) excess across names, each name counted only inside its own
    traded window (before listing / after death it must not dilute the mean), plus the
    per-name total-excess table."""
    ret = close.pct_change(fill_method=None)
    target = box_target(close, window=BOX_WINDOW).shift(1).fillna(1.0)   # execute at t+1
    res = fractional_target_backtest(target, ret, costs=costs)
    # A name counts between its first and last real close (suspension gaps included: the
    # position is held through them); outside that window it must not dilute the mean.
    has = close.notna()
    alive = has.cummax() & has.iloc[::-1].cummax().iloc[::-1]
    diff = (res["net"] - ret.fillna(0.0)).where(alive)
    per_name = pd.DataFrame({"excess": diff.sum(), "traded": res["traded"].sum(),
                             "days": alive.sum()})
    return diff.mean(axis=1).dropna(), per_name


def _line(tag: str, diff: pd.Series) -> dict:
    return {"tag": tag, "ann_excess": float(diff.mean() * 252),
            "sharpe_diff": sharpe(diff),
            "t_month": clustered_tstat(diff, diff.index, freq="M")}


def _print(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'variant':>30} {'excess/yr':>10} {'Sharpe(diff)':>13} {'t(month)':>9}")
    for r in rows:
        print(f"  {r['tag']:>30} {r['ann_excess']:>+10.2%} {r['sharpe_diff']:>13.2f} "
              f"{r['t_month']:>9.2f}")


def main() -> None:
    ensure_dirs()
    hs = pd.read_parquet(MEMBERSHIP_PARQUET)
    cs = pd.read_parquet(CSI500_MEMBERSHIP_PARQUET)
    union = sorted(set(hs["code"]) | set(cs["code"]))
    buckets = pd.read_parquet(INDUSTRY_PARQUET).set_index("code")["bucket"]

    close_all = load_close_panel(codes=union, field="close")
    quality_ever = quality_mask(close_all.index, close_all.columns).any()
    primary = [c for c in union if quality_ever.get(c, False)
               and pd.notna(buckets.get(c)) and buckets.get(c) not in CYCLICAL_BUCKETS]
    print(f"primary set: {len(primary)} non-cyclical quality names "
          f"(ever pass the 3x annual ROE > 15% gate, in the frozen buckets)")

    close = close_all[primary]
    diff_net, per_name = excess_series(close, costs=AShareCosts())
    diff_gross, _ = excess_series(close, costs=ZERO_COSTS)

    rows = [_line("primary NET", diff_net), _line("primary GROSS", diff_gross)]
    for s, e in ERAS:
        sl = slice(pd.Timestamp(s), pd.Timestamp(e))
        rows.append(_line(f"primary NET {s[:4]}-{e[:4]}", diff_net.loc[sl]))
    hs300 = sorted(set(hs["code"]))
    diff_hs, _ = excess_series(close_all[hs300], costs=AShareCosts())
    rows.append(_line("all PIT HS300 names NET", diff_hs))
    _print(rows, "120-DAY BOX, HALF POSITION vs BUY-AND-HOLD:")

    print(f"\nper-name: hit rate {(per_name['excess'] > 0).mean():.1%} "
          f"({int((per_name['excess'] > 0).sum())}/{len(per_name)} names beat holding); "
          f"mean total traded {per_name['traded'].mean():.1f}x of account value")
    drag = diff_gross.mean() * 252 - diff_net.mean() * 252
    print(f"friction drag: {drag:+.2%}/yr of the gross-vs-net gap")

    head = rows[0]
    verdict = head["ann_excess"] > 0 and head["t_month"] > 2
    print(f"\nVERDICT: {'CONFIRMED' if verdict else 'REJECTED'} -- needs net excess > 0 with "
          f"monthly-clustered t > 2 (got {head['ann_excess']:+.2%}/yr, t {head['t_month']:.2f})")

    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "box_trading_summary.parquet",
                      index=False)
    atomic_to_parquet(per_name.reset_index(names="code"),
                      BACKTESTS_DIR / "box_trading_per_name.parquet", index=False)
    print(f"saved -> {BACKTESTS_DIR / 'box_trading_summary.parquet'}")


if __name__ == "__main__":
    main()
