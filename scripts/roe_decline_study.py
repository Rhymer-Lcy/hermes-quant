"""The ROE-decline exit: is the sell trigger worth acting on? Issue #11.

The friend's growth-stock exit rule, pre-registered before this script existed: a marked
annual-ROE decline at a quality name means development may be constrained -- get out.

  event      publication day of an annual report printing >= 5pp below the mean of the
             three previously published annual ROEs, where that prior mean was >= 15%
  universe   PIT HS300+CSI500 members, non-ST, >= 20 prior traded days, events 2015+
  entry      the close of the first trading day after publication (the sell moment)
  verdict    CONFIRMED only if the 250d net abnormal mean is NEGATIVE with monthly-clustered
             t < -2 (costs charged AGAINST the sell reading: +0.20% added to the drift)

    conda activate hermes
    python scripts/roe_decline_study.py
"""
from __future__ import annotations

import pandas as pd

from hermes.data.fundamentals import load_annual_roe
from hermes.data.lake import load_close_panel
from hermes.data.membership import (CSI500_MEMBERSHIP_PARQUET, MEMBERSHIP_PARQUET,
                                    membership_lookup)
from hermes.io import atomic_to_parquet
from hermes.paths import BACKTESTS_DIR, ensure_dirs
from hermes.research.backtest.limit_events import event_cars
from hermes.research.backtest.metrics import clustered_tstat, tstat

DROP_PP = 0.05
QUALITY_MEAN = 0.15
HORIZONS = (20, 60, 120, 250)
VERDICT_HORIZON = 250
RT_COST = 0.0020
ERAS = [("2015-01-01", "2019-12-31"), ("2020-01-01", "2026-12-31")]


def find_events() -> pd.DataFrame:
    roe = load_annual_roe()
    rows = []
    for code, grp in roe.groupby("code"):
        g = grp.sort_values("pubDate").reset_index(drop=True)
        for i in range(3, len(g)):
            prior = g["roeAvg"].iloc[i - 3:i].mean()
            drop = prior - g["roeAvg"].iloc[i]
            if prior >= QUALITY_MEAN and drop >= DROP_PP:
                rows.append({"code": code, "pub_date": g["pubDate"].iloc[i],
                             "prior_mean": float(prior),
                             "new_roe": float(g["roeAvg"].iloc[i]), "drop": float(drop)})
    return pd.DataFrame(rows)


def _report(tag: str, cars: pd.DataFrame, col: str) -> dict:
    x = cars[col]
    return {"tag": tag, "n": int(x.notna().sum()), "mean": float(x.mean()),
            "median": float(x.median()), "t_event": tstat(x),
            "t_month": clustered_tstat(x, cars["entry_date"], freq="M"),
            "hit": float((x < 0).mean())}


def _print(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'sample':>26} {'N':>5} {'mean':>8} {'median':>8} {'t(event)':>9} "
          f"{'t(month)':>9} {'neg':>6}")
    for r in rows:
        print(f"  {r['tag']:>26} {r['n']:>5} {r['mean']:>+8.2%} {r['median']:>+8.2%} "
              f"{r['t_event']:>9.2f} {r['t_month']:>9.2f} {r['hit']:>6.1%}")


def main() -> None:
    ensure_dirs()
    ev = find_events()
    hs = pd.read_parquet(MEMBERSHIP_PARQUET)
    cs = pd.read_parquet(CSI500_MEMBERSHIP_PARQUET)
    union = sorted(set(hs["code"]) | set(cs["code"]))
    close = load_close_panel(codes=union, field="close")
    st = load_close_panel(codes=union, field="isST")
    hs_asof, cs_asof = membership_lookup(hs), membership_lookup(cs)
    member = pd.DataFrame({c: [c in hs_asof(d) or c in cs_asof(d) for d in close.index]
                           for c in close.columns}, index=close.index)
    universe = (member & ~st.eq(True) & close.notna()
                & (close.notna().cumsum().shift(1) >= 20))
    ret = close.pct_change(fill_method=None)
    abn = ret.sub(ret.where(universe).mean(axis=1), axis=0)

    idx = close.index
    ev = ev[ev["pub_date"] >= idx.min()]
    pos = idx.searchsorted(ev["pub_date"], side="right")
    ev = ev[pos < len(idx)].copy()
    ev["entry_date"] = idx[idx.searchsorted(ev["pub_date"], side="right")]
    keep = [c in universe.columns and bool(universe.loc[d, c])
            for c, d in zip(ev["code"], ev["entry_date"])]
    ev = ev[pd.Series(keep, index=ev.index)].reset_index(drop=True)
    print(f"events: {len(ev)} marked ROE declines at quality names in "
          f"{ev['code'].nunique()} names, {ev['pub_date'].dt.year.min()} -> "
          f"{ev['pub_date'].dt.year.max()}")

    cars = event_cars(ev, abn, HORIZONS, rt_cost=0.0)
    for h in HORIZONS:
        cars[f"car_{h}_net"] = cars[f"car_{h}"] + RT_COST   # conservative for a SELL rule

    rows = [_report(f"{h}d net", cars, f"car_{h}_net") for h in HORIZONS]
    _print(rows, "AFTER A PUBLISHED ROE DECLINE AT A QUALITY NAME (abnormal drift):")

    v = f"car_{VERDICT_HORIZON}_net"
    tri = pd.qcut(cars["drop"], 3, labels=["mild", "middle", "severe"])
    subs = [_report(f"decline {lab}", cars[tri == lab], v)
            for lab in ("mild", "middle", "severe")]
    subs.append(_report("still >= 15% after", cars[cars["new_roe"] >= QUALITY_MEAN], v))
    subs.append(_report("broken below 15%", cars[cars["new_roe"] < QUALITY_MEAN], v))
    for s, e in ERAS:
        cut = cars[(cars["pub_date"] >= s) & (cars["pub_date"] <= e)]
        subs.append(_report(f"{s[:4]}-{e[:4]}", cut, v))
    _print(subs, f"pre-registered sub-samples at {VERDICT_HORIZON}d:")

    head = rows[HORIZONS.index(VERDICT_HORIZON)]
    verdict = head["mean"] < 0 and head["t_month"] < -2
    print(f"\nVERDICT: {'CONFIRMED' if verdict else 'REJECTED'} -- the exit rule needs "
          f"{VERDICT_HORIZON}d net abnormal < 0 with monthly-clustered t < -2 "
          f"(got {head['mean']:+.2%}, t {head['t_month']:.2f})")

    atomic_to_parquet(cars, BACKTESTS_DIR / "roe_decline_cars.parquet", index=False)
    atomic_to_parquet(pd.DataFrame(rows + subs),
                      BACKTESTS_DIR / "roe_decline_summary.parquet", index=False)
    print(f"saved -> {BACKTESTS_DIR / 'roe_decline_summary.parquet'}")


if __name__ == "__main__":
    main()
