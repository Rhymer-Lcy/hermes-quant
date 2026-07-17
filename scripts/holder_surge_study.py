"""A quarterly shareholder-count surge as a distribution signal. Issue #13.

The friend's sell trigger, pre-registered before the holder-count lake existed: if the
shareholder count jumps ~20% within one quarter, the chips may have been handed to retail
-- consider selling. NARI Technology (sh.600406) is his illustration.

  event      a disclosure whose count rose >= 20% vs the previously disclosed count, with
             the two stat dates <= 120 calendar days apart; PIT anchor = ANNOUNCEMENT date
  universe   PIT HS300+CSI500 members, non-ST, >= 20 prior traded days, events 2015+
  entry      close of the first trading day after the announcement (the sell moment)
  verdict    CONFIRMED only if the 250d net abnormal mean is NEGATIVE with monthly-clustered
             t < -2 (costs charged AGAINST the sell reading: +0.20% added to the drift)

    conda activate hermes
    python scripts/holder_surge_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.data.fundamentals import load_holder_counts
from hermes.data.lake import load_close_panel
from hermes.data.membership import (CSI500_MEMBERSHIP_PARQUET, MEMBERSHIP_PARQUET,
                                    membership_lookup)
from hermes.io import atomic_to_parquet
from hermes.paths import BACKTESTS_DIR, ensure_dirs
from hermes.research.backtest.limit_events import event_cars
from hermes.research.backtest.metrics import clustered_tstat, tstat

SURGE = 0.20
MAX_GAP_DAYS = 120
HORIZONS = (20, 60, 120, 250)
VERDICT_HORIZON = 250
RT_COST = 0.0020
EXAMPLE = "sh.600406"          # the friend's illustration; descriptive annex only
ERAS = [("2015-01-01", "2019-12-31"), ("2020-01-01", "2026-12-31")]


def find_events() -> pd.DataFrame:
    hc = load_holder_counts()
    rows = []
    for code, grp in hc.groupby("code"):
        g = grp.sort_values("stat_date").reset_index(drop=True)
        for i in range(1, len(g)):
            prev, cur = g["holders"].iloc[i - 1], g["holders"].iloc[i]
            gap = (g["stat_date"].iloc[i] - g["stat_date"].iloc[i - 1]).days
            if prev > 0 and gap <= MAX_GAP_DAYS and cur / prev - 1.0 >= SURGE:
                rows.append({"code": code, "stat_date": g["stat_date"].iloc[i],
                             "prev_stat_date": g["stat_date"].iloc[i - 1],
                             "pub_date": g["pub_date"].iloc[i],
                             "surge": float(cur / prev - 1.0)})
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
    print(f"events: {len(ev)} quarterly holder-count surges >= {SURGE:.0%} in "
          f"{ev['code'].nunique()} names, {ev['pub_date'].dt.year.min()} -> "
          f"{ev['pub_date'].dt.year.max()}")

    # The stock's own return between the two stat dates, from lake closes (frozen
    # sub-sample: the distribution narrative implies the surge rode a rally).
    cf = close.ffill()
    colpos = {c: j for j, c in enumerate(cf.columns)}
    ival = []
    for c, d0, d1 in zip(ev["code"], ev["prev_stat_date"], ev["stat_date"]):
        j = colpos.get(c)
        i0, i1 = idx.searchsorted(d0, side="right") - 1, idx.searchsorted(d1, side="right") - 1
        if j is None or i0 < 0 or i1 <= i0:
            ival.append(np.nan)
            continue
        p0, p1 = cf.iat[i0, j], cf.iat[i1, j]
        ival.append(p1 / p0 - 1.0 if p0 > 0 and not (np.isnan(p0) or np.isnan(p1))
                    else np.nan)
    ev["ival_ret"] = ival

    cars = event_cars(ev, abn, HORIZONS, rt_cost=0.0)
    for h in HORIZONS:
        cars[f"car_{h}_net"] = cars[f"car_{h}"] + RT_COST   # conservative for a SELL rule

    rows = [_report(f"{h}d net", cars, f"car_{h}_net") for h in HORIZONS]
    _print(rows, "AFTER A PUBLISHED QUARTERLY HOLDER-COUNT SURGE (abnormal drift):")

    v = f"car_{VERDICT_HORIZON}_net"
    tri = pd.qcut(cars["surge"], 3, labels=["mild", "middle", "severe"])
    subs = [_report(f"surge {lab}", cars[tri == lab], v)
            for lab in ("mild", "middle", "severe")]
    subs.append(_report("interval rally", cars[cars["ival_ret"] > 0], v))
    subs.append(_report("interval decline", cars[cars["ival_ret"] <= 0], v))
    for s, e in ERAS:
        cut = cars[(cars["pub_date"] >= s) & (cars["pub_date"] <= e)]
        subs.append(_report(f"{s[:4]}-{e[:4]}", cut, v))
    _print(subs, f"pre-registered sub-samples at {VERDICT_HORIZON}d:")

    # Descriptive annex (no verdict weight): the friend's single illustration.
    ann = cars[cars["code"] == EXAMPLE]
    print(f"\nthe illustration ({EXAMPLE}): {len(ann)} surge events"
          + ("" if ann.empty else " --"))
    for _, r in ann.iterrows():
        tail = "window open" if pd.isna(r[v]) else f"250d net {r[v]:+.2%}"
        print(f"  {r['stat_date'].date()} surge {r['surge']:+.1%} "
              f"(interval {r['ival_ret']:+.1%}) -> {tail}")

    head = rows[HORIZONS.index(VERDICT_HORIZON)]
    verdict = head["mean"] < 0 and head["t_month"] < -2
    print(f"\nVERDICT: {'CONFIRMED' if verdict else 'REJECTED'} -- the sell trigger needs "
          f"{VERDICT_HORIZON}d net abnormal < 0 with monthly-clustered t < -2 "
          f"(got {head['mean']:+.2%}, t {head['t_month']:.2f})")

    atomic_to_parquet(cars, BACKTESTS_DIR / "holder_surge_cars.parquet", index=False)
    atomic_to_parquet(pd.DataFrame(rows + subs),
                      BACKTESTS_DIR / "holder_surge_summary.parquet", index=False)
    print(f"saved -> {BACKTESTS_DIR / 'holder_surge_summary.parquet'}")


if __name__ == "__main__":
    main()
