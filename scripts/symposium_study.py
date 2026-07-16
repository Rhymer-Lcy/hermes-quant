"""Does speaking at the Premier's economic symposium mark a stock to buy? Issue #7.

The friend's policy-selection proxy, pre-registered before the event list was compiled: the
Premier's roughly-quarterly 经济形势专家和企业家座谈会 names its entrepreneur speakers in the
official report; the claim is that an invitation marks the company as policy-favored.

  events     every (symposium, entrepreneur speaker) pair 2015 -> present, from the official
             report; the event list is hand-compiled from public archives with one source URL
             per event and lives TRACKED in data/manual/symposium_events.csv (the one input a
             study cannot regenerate from a vendor)
  entry      the close of the first trading day after the report date
  abnormal   name return minus the same-day EW mean of PIT HS300 members, non-ST (the
             size-appropriate benchmark -- attendees are almost all large caps; the sibling
             13F study's size-confound lesson applied prospectively); full-universe EW
             abnormal reported as texture
  verdict    CONFIRMED only if the 250-trading-day net abnormal mean is positive with
             monthly-clustered t > 2 (costs 0.20% round trip)

    conda activate hermes
    python scripts/symposium_study.py
"""
from __future__ import annotations

import pandas as pd

from hermes.data.lake import load_close_panel
from hermes.data.membership import (CSI500_MEMBERSHIP_PARQUET, MEMBERSHIP_PARQUET,
                                    membership_lookup)
from hermes.io import atomic_to_parquet
from hermes.paths import BACKTESTS_DIR, DATA_DIR, ensure_dirs
from hermes.research.backtest.limit_events import event_cars
from hermes.research.backtest.metrics import clustered_tstat, tstat

EVENTS_CSV = DATA_DIR / "manual" / "symposium_events.csv"
HORIZONS = (5, 20, 60, 120, 250)
VERDICT_HORIZON = 250
RT_COST = 0.0020
MIN_HISTORY = 20
LI_QIANG_FROM = pd.Timestamp("2023-03-11")


def abnormal_panels(extra_codes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(abn_hs300, abn_universe, close): abnormal-return panels against the two frozen
    benchmarks -- the EW PIT HS300 member mean (the verdict benchmark) and the EW PIT
    HS300+CSI500 union mean (the texture benchmark). `extra_codes` are event names outside
    the membership union (pulled into the lake for this study); they get abnormal returns
    but can never enter a benchmark mean."""
    hs = pd.read_parquet(MEMBERSHIP_PARQUET)
    cs = pd.read_parquet(CSI500_MEMBERSHIP_PARQUET)
    union = sorted(set(hs["code"]) | set(cs["code"]) | set(extra_codes))
    close = load_close_panel(codes=union, field="close")
    st = load_close_panel(codes=union, field="isST")
    hs_asof, cs_asof = membership_lookup(hs), membership_lookup(cs)
    ok = ~st.eq(True) & close.notna() & (close.notna().cumsum().shift(1) >= MIN_HISTORY)
    hs_member = pd.DataFrame({c: [c in hs_asof(d) for d in close.index]
                              for c in close.columns}, index=close.index)
    cs_member = pd.DataFrame({c: [c in cs_asof(d) for d in close.index]
                              for c in close.columns}, index=close.index)
    ret = close.pct_change(fill_method=None)
    bench_hs = ret.where(hs_member & ok).mean(axis=1)
    bench_uni = ret.where((hs_member | cs_member) & ok).mean(axis=1)
    return ret.sub(bench_hs, axis=0), ret.sub(bench_uni, axis=0), close


def _report(tag: str, cars: pd.DataFrame, col: str) -> dict:
    x = cars[col]
    return {"tag": tag, "n": int(x.notna().sum()), "mean": float(x.mean()),
            "median": float(x.median()), "t_event": tstat(x),
            "t_month": clustered_tstat(x, cars["entry_date"], freq="M"),
            "hit": float((x > 0).mean())}


def _print(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'sample':>28} {'N':>4} {'mean':>8} {'median':>8} {'t(event)':>9} "
          f"{'t(month)':>9} {'hit':>6}")
    for r in rows:
        print(f"  {r['tag']:>28} {r['n']:>4} {r['mean']:>+8.2%} {r['median']:>+8.2%} "
              f"{r['t_event']:>9.2f} {r['t_month']:>9.2f} {r['hit']:>6.1%}")


def main() -> None:
    ensure_dirs()
    raw = pd.read_csv(EVENTS_CSV)
    raw["meeting_date"] = pd.to_datetime(raw["meeting_date"])
    raw["report_date"] = pd.to_datetime(raw["report_date"])
    mapped = raw[raw["mapping"].isin(["direct", "flagship"])].copy()
    print(f"event list: {len(raw)} entrepreneur speeches at "
          f"{raw['meeting_date'].nunique()} symposiums; {len(mapped)} map to A-share tickers "
          f"({(raw['mapping'] == 'excluded-unlisted').sum()} unlisted, "
          f"{(raw['mapping'] == 'excluded-hkus').sum()} HK/US-only excluded)")

    abn_hs, abn_uni, close = abnormal_panels(sorted(mapped["code"].astype(str).unique()))
    idx = close.index
    entry_pos = idx.searchsorted(mapped["report_date"], side="right")
    keep = entry_pos < len(idx)
    mapped = mapped[keep].copy()
    mapped["entry_date"] = idx[entry_pos[keep]]
    mapped["code"] = mapped["code"].astype(str)
    missing = [c for c in mapped["code"].unique() if c not in close.columns]
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} mapped codes missing from the lake: {missing}")

    cars = event_cars(mapped[["code", "entry_date", "meeting_date"]], abn_hs, HORIZONS,
                      rt_cost=0.0)
    cars_uni = event_cars(mapped[["code", "entry_date"]], abn_uni, (VERDICT_HORIZON,),
                          rt_cost=0.0)
    for h in HORIZONS:
        cars[f"car_{h}_net"] = cars[f"car_{h}"] - RT_COST

    rows = [_report(f"{h}d net (vs HS300 EW)", cars, f"car_{h}_net") for h in HORIZONS]
    _print(rows, "SYMPOSIUM SPEAKERS -- buy the close after the official report:")

    v = f"car_{VERDICT_HORIZON}_net"
    first = ~cars["code"].duplicated()
    subs = [_report("first-ever appearance", cars[first], v),
            _report("repeat appearance", cars[~first], v),
            _report("2015-2019", cars[cars["meeting_date"] < "2020-01-01"], v),
            _report("2020-2026", cars[cars["meeting_date"] >= "2020-01-01"], v),
            _report("Li Keqiang era", cars[cars["meeting_date"] < LI_QIANG_FROM], v),
            _report("Li Qiang era", cars[cars["meeting_date"] >= LI_QIANG_FROM], v)]
    cars_uni[f"car_{VERDICT_HORIZON}_net"] = cars_uni[f"car_{VERDICT_HORIZON}"] - RT_COST
    subs.append(_report("texture: vs universe EW", cars_uni, v))
    _print(subs, f"pre-registered sub-samples at {VERDICT_HORIZON}d:")

    head = rows[HORIZONS.index(VERDICT_HORIZON)]
    verdict = head["mean"] > 0 and head["t_month"] > 2
    print(f"\nVERDICT: {'CONFIRMED' if verdict else 'REJECTED'} -- needs {VERDICT_HORIZON}d "
          f"net abnormal > 0 with monthly-clustered t > 2 "
          f"(got {head['mean']:+.2%}, t {head['t_month']:.2f})")

    atomic_to_parquet(cars, BACKTESTS_DIR / "symposium_cars.parquet", index=False)
    atomic_to_parquet(pd.DataFrame(rows + subs), BACKTESTS_DIR / "symposium_summary.parquet",
                      index=False)
    print(f"saved -> {BACKTESTS_DIR / 'symposium_summary.parquet'}")


if __name__ == "__main__":
    main()
