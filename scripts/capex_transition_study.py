"""Buying the growth-to-dividend transition: capex past peak, payout formed. Issue #14.

The friend's lifecycle rule, pre-registered before the capex lake existed: the growth
ceiling is the turn where capex winds down and the name becomes a dividend stock (China
Shenhua "the classic") -- BUY that transition and let the dividends run.

  event      first annual report per name with >= 5 prior positive-capex reports, where
             capex <= 60% of its prior-5-report peak AND trailing yield >= 3% at the
             publication date (NOTICE_DATE, the point-in-time anchor)
  universe   PIT HS300+CSI500 members, non-ST, >= 20 prior traded days, events 2015+
  entry      close of the first trading day after publication; one flat retail round trip
             charged against the BUY reading
  verdict    CONFIRMED only if the 250d net abnormal mean is POSITIVE with
             monthly-clustered t > 2

    conda activate hermes
    python scripts/capex_transition_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.data.fundamentals import (DIVIDENDS_PARQUET, RAW_CLOSE_PARQUET, load_capex,
                                      trailing_yield)
from hermes.data.lake import load_close_panel
from hermes.data.membership import (CSI500_MEMBERSHIP_PARQUET, MEMBERSHIP_PARQUET,
                                    membership_lookup)
from hermes.io import atomic_to_parquet
from hermes.paths import BACKTESTS_DIR, ensure_dirs
from hermes.research.backtest.limit_events import event_cars
from hermes.research.backtest.metrics import clustered_tstat, tstat

PEAK_RATIO = 0.60
YIELD_GATE = 0.03
MIN_PRIOR_REPORTS = 5
HORIZONS = (20, 60, 120, 250, 500)
VERDICT_HORIZON = 250
RT_COST = 0.0020
EXAMPLE = "sh.601088"          # China Shenhua, the friend's classic; descriptive annex
ERAS = [("2015-01-01", "2019-12-31"), ("2020-01-01", "2026-12-31")]


def find_events(cap: pd.DataFrame, yld: pd.DataFrame, ratio: float = PEAK_RATIO,
                ygate: float = YIELD_GATE) -> pd.DataFrame:
    """First qualifying transition report per name (the lifecycle happens once)."""
    yidx, ycol = yld.index, {c: j for j, c in enumerate(yld.columns)}

    def yld_at(code: str, d: pd.Timestamp) -> float:
        j = ycol.get(code)
        p = yidx.searchsorted(d, side="right") - 1
        return float(yld.iat[p, j]) if j is not None and p >= 0 else np.nan

    rows = []
    for code, grp in cap.groupby("code"):
        g = grp.dropna(subset=["capex"]).sort_values("stat_date").reset_index(drop=True)
        for i in range(MIN_PRIOR_REPORTS, len(g)):
            prior = g["capex"].iloc[i - MIN_PRIOR_REPORTS:i]
            if not (prior > 0).all() or g["capex"].iloc[i] > ratio * prior.max():
                continue
            y = yld_at(code, g["pub_date"].iloc[i])
            if not np.isnan(y) and y >= ygate:
                rows.append({"code": code, "stat_date": g["stat_date"].iloc[i],
                             "pub_date": g["pub_date"].iloc[i],
                             "capex_ratio": float(g["capex"].iloc[i] / prior.max()),
                             "yield_at": float(y)})
                break
    return pd.DataFrame(rows)


def _report(tag: str, cars: pd.DataFrame, col: str) -> dict:
    x = cars[col]
    return {"tag": tag, "n": int(x.notna().sum()), "mean": float(x.mean()),
            "median": float(x.median()), "t_event": tstat(x),
            "t_month": clustered_tstat(x, cars["entry_date"], freq="M"),
            "pos": float((x > 0).mean())}


def _print(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'sample':>26} {'N':>5} {'mean':>8} {'median':>8} {'t(event)':>9} "
          f"{'t(month)':>9} {'pos':>6}")
    for r in rows:
        print(f"  {r['tag']:>26} {r['n']:>5} {r['mean']:>+8.2%} {r['median']:>+8.2%} "
              f"{r['t_event']:>9.2f} {r['t_month']:>9.2f} {r['pos']:>6.1%}")


def _attach(ev: pd.DataFrame, idx: pd.DatetimeIndex, universe: pd.DataFrame) -> pd.DataFrame:
    ev = ev[(ev["pub_date"] >= idx.min())
            & (idx.searchsorted(ev["pub_date"], side="right") < len(idx))].copy()
    ev["entry_date"] = idx[idx.searchsorted(ev["pub_date"], side="right")]
    keep = [c in universe.columns and bool(universe.loc[d, c])
            for c, d in zip(ev["code"], ev["entry_date"])]
    return ev[pd.Series(keep, index=ev.index)].reset_index(drop=True)


def main() -> None:
    ensure_dirs()
    cap = load_capex()
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

    div = pd.read_parquet(DIVIDENDS_PARQUET)
    div["ex_date"] = pd.to_datetime(div["ex_date"])
    raw = pd.read_parquet(RAW_CLOSE_PARQUET)
    yld = trailing_yield(div, raw)

    ev = _attach(find_events(cap, yld), idx, universe)
    print(f"events: {len(ev)} transitions (capex <= {PEAK_RATIO:.0%} of prior-5 peak, "
          f"yield >= {YIELD_GATE:.0%}), {ev['pub_date'].dt.year.min()} -> "
          f"{ev['pub_date'].dt.year.max()}")

    cars = event_cars(ev, abn, HORIZONS, rt_cost=RT_COST)
    rows = [_report(f"{h}d net", cars, f"car_{h}") for h in HORIZONS]
    _print(rows, "AFTER A PUBLISHED GROWTH-TO-DIVIDEND TRANSITION (net abnormal drift):")

    v = f"car_{VERDICT_HORIZON}"
    subs = []
    for lab, grp in cars.groupby(pd.qcut(cars["capex_ratio"], 3,
                                         labels=["deep", "middle", "shallow"]),
                                 observed=True):
        subs.append(_report(f"capex cut {lab}", grp, v))
    for lab, grp in cars.groupby(pd.qcut(cars["yield_at"], 3,
                                         labels=["low", "mid", "high"]),
                                 observed=True):
        subs.append(_report(f"yield {lab}", grp, v))
    for s, e in ERAS:
        subs.append(_report(f"{s[:4]}-{e[:4]}",
                            cars[(cars["pub_date"] >= s) & (cars["pub_date"] <= e)], v))
    _print(subs, f"pre-registered sub-samples at {VERDICT_HORIZON}d:")

    vrows = []
    for r in (0.50, 0.70):
        c = event_cars(_attach(find_events(cap, yld, ratio=r), idx, universe),
                       abn, (VERDICT_HORIZON,), rt_cost=RT_COST)
        vrows.append(_report(f"peak ratio {r:.0%}", c, v))
    for yg in (0.02, 0.04):
        c = event_cars(_attach(find_events(cap, yld, ygate=yg), idx, universe),
                       abn, (VERDICT_HORIZON,), rt_cost=RT_COST)
        vrows.append(_report(f"yield gate {yg:.0%}", c, v))
    _print(vrows, "sensitivity variants (reported, never used for selection):")

    # Descriptive annex 1: the friend's classic.
    ann = cars[cars["code"] == EXAMPLE]
    if ann.empty:
        print(f"\nthe classic ({EXAMPLE}): no qualifying transition under the frozen marker")
    else:
        for _, r in ann.iterrows():
            t500 = "window open" if pd.isna(r["car_500"]) else f"{r['car_500']:+.2%}"
            print(f"\nthe classic ({EXAMPLE}): transition report {r['stat_date'].date()} "
                  f"(capex at {r['capex_ratio']:.0%} of peak, yield {r['yield_at']:.1%}) -> "
                  f"250d net {r[v]:+.2%}, 500d {t500}")

    # Descriptive annex 2 (frozen as descriptive; no verdict weight): the payback clock.
    rawidx = raw.index
    payback = []
    for _, r in cars.iterrows():
        code = r["code"]
        if code not in raw.columns:
            continue
        p = rawidx.searchsorted(r["entry_date"], side="left")
        if p >= len(rawidx) or np.isnan(raw[code].iloc[p]):
            continue
        cost = float(raw[code].iloc[p])
        dps = (div[(div["code"] == code) & (div["ex_date"] > r["entry_date"])]
               .sort_values("ex_date"))
        cum = dps["dps"].cumsum()
        hit = cum[cum >= cost]
        if not hit.empty:
            pb_date = dps.loc[hit.index[0], "ex_date"]
            yrs = (pb_date - r["entry_date"]).days / 365.25
            q = idx.searchsorted(pb_date, side="right")
            drift = (abn[code].iloc[q:q + 250].sum()
                     if q + 250 < len(idx) and code in abn.columns else np.nan)
            payback.append({"code": code, "years": yrs, "post_drift": drift})
    pb = pd.DataFrame(payback)
    print(f"\npayback clock (descriptive, unreinvested DPS >= entry price): "
          f"{len(pb)} of {len(cars)} events paid back in-window"
          + ("" if pb.empty else
             f"; median {pb['years'].median():.1f} yrs; post-payback 250d drift "
             f"{pb['post_drift'].mean():+.2%} (N closed {int(pb['post_drift'].notna().sum())})"))

    head = rows[HORIZONS.index(VERDICT_HORIZON)]
    verdict = head["mean"] > 0 and head["t_month"] > 2
    print(f"\nVERDICT: {'CONFIRMED' if verdict else 'REJECTED'} -- buying the transition "
          f"needs {VERDICT_HORIZON}d net abnormal > 0 with monthly-clustered t > 2 "
          f"(got {head['mean']:+.2%}, t {head['t_month']:.2f})")

    atomic_to_parquet(cars, BACKTESTS_DIR / "capex_transition_cars.parquet", index=False)
    atomic_to_parquet(pd.DataFrame(rows + subs + vrows),
                      BACKTESTS_DIR / "capex_transition_summary.parquet", index=False)
    print(f"saved -> {BACKTESTS_DIR / 'capex_transition_summary.parquet'}")


if __name__ == "__main__":
    main()
