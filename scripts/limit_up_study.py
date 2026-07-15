"""After a limit-up seal, is there anything left to buy? Pre-registered as issue #1.

The A-share twin of the sibling plutus gap studies: there, any US stock that gaps up >= +20%
overnight bleeds ~-3% abnormal over the next month. A-share price limits FORBID that gap and
smear the move across locked sessions, and two priors collide: lottery/attention (the entry
bleeds) vs truncated price discovery (the 打板 thesis: what remains is continuation). Design
frozen in the issue BEFORE this script existed:

  universe   PIT HS300+CSI500 members, non-ST, >= 20 prior traded days, 2015 -> present
             (disclosed: the folklore's micro-cap home turf is OUTSIDE these indices; this is
             the investable version)
  event      a FRESH limit-up-locked close (date-aware widths; the detection bug that applied
             today's rules to history was fixed and unit-tested first)
  entry      the first later close NOT limit-locked -- the first fillable EOD print; the
             abnormal return earned while sealed is recorded as the part you cannot get
  horizons   1/5/10/20/60 trading days from entry, abnormal vs the same-day equal-weight PIT
             member (non-ST) mean; costs 0.20% round trip
  inference  clustering-robust t on MONTHLY means (2015 alone is ~29% of events)
  verdict    CONFIRMED only if the 20d net mean is positive with monthly-clustered t > 2

Context leg (reported, not the verdict): fresh limit-DOWN closes, same conventions -- the
"catching the falling knife" read; not harvestable short (no retail shorting in A-shares).

    conda activate hermes
    python scripts/limit_up_study.py
"""
from __future__ import annotations

import pandas as pd
from scipy import stats

from hermes.data.ingest import board_of
from hermes.data.lake import load_close_panel
from hermes.data.membership import (CSI500_MEMBERSHIP_PARQUET, MEMBERSHIP_PARQUET,
                                    membership_lookup)
from hermes.io import atomic_to_parquet
from hermes.paths import BACKTESTS_DIR, ensure_dirs
from hermes.research.backtest.limit_events import event_cars, fresh_events, resolve_entries
from hermes.research.backtest.limits import limit_flags, limit_pct_panel
from hermes.research.backtest.metrics import clustered_tstat, tstat

HORIZONS = (1, 5, 10, 20, 60)
VERDICT_HORIZON = 20
RT_COST = 0.0020            # the repo's documented retail round trip (index_effect_study)
MIN_HISTORY = 20
ERAS = [("2015-01-01", "2019-12-31"), ("2020-01-01", "2026-12-31")]


def _report(tag: str, cars: pd.DataFrame, col: str) -> dict:
    x = cars[col]
    return {"tag": tag, "n": int(x.notna().sum()), "mean": float(x.mean()),
            "median": float(x.median()), "t_event": tstat(x),
            "t_month": clustered_tstat(x, cars["date"], freq="M"),
            "hit": float((x > 0).mean())}


def _print(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'sample':>24} {'N':>6} {'mean':>8} {'median':>8} {'t(event)':>9} "
          f"{'t(month)':>9} {'hit':>6}")
    for r in rows:
        print(f"  {r['tag']:>24} {r['n']:>6} {r['mean']:>+8.2%} {r['median']:>+8.2%} "
              f"{r['t_event']:>9.2f} {r['t_month']:>9.2f} {r['hit']:>6.1%}")


def main() -> None:
    ensure_dirs()
    hs = pd.read_parquet(MEMBERSHIP_PARQUET)
    cs = pd.read_parquet(CSI500_MEMBERSHIP_PARQUET)
    union = sorted(set(hs["code"]) | set(cs["code"]))
    close = load_close_panel(codes=union, field="close")
    pre = load_close_panel(codes=union, field="preclose")
    st = load_close_panel(codes=union, field="isST")
    hs_asof, cs_asof = membership_lookup(hs), membership_lookup(cs)

    # PIT member mask (monthly snapshots, as-of) and the benchmark universe: member, non-ST.
    hs_sets = {d: hs_asof(d) for d in close.index}
    cs_sets = {d: cs_asof(d) for d in close.index}
    member = pd.DataFrame({c: [c in hs_sets[d] or c in cs_sets[d] for d in close.index]
                           for c in close.columns}, index=close.index)
    not_st = ~st.eq(True)
    universe = member & not_st & close.notna()

    ret = close.pct_change(fill_method=None)
    abn = ret.sub(ret.where(universe).mean(axis=1), axis=0)
    flags = limit_flags(close, pre)
    history_ok = close.notna().cumsum().shift(1) >= MIN_HISTORY

    def build(direction: int) -> tuple[pd.DataFrame, int]:
        ev = fresh_events(flags, direction=direction)
        keep = [bool(universe.at[d, c]) and bool(history_ok.at[d, c])
                for c, d in zip(ev["code"], ev["date"])]
        ev = ev[pd.Series(keep, index=ev.index)].reset_index(drop=True)
        ev = resolve_entries(ev, flags, abn, direction=direction)
        unresolved = int(ev["entry_date"].isna().sum())
        out = event_cars(ev, abn, HORIZONS, rt_cost=0.0)          # gross; net added below
        for h in HORIZONS:
            out[f"car_{h}_net"] = out[f"car_{h}"] - RT_COST
        return out, unresolved

    cars, n_unresolved = build(direction=1)
    print(f"universe: {len(union)} names ever (PIT HS300+CSI500), "
          f"{close.index.min().date()} -> {close.index.max().date()}")
    print(f"events: {len(cars):,} fresh limit-up-locked closes in {cars['code'].nunique():,} "
          f"names; unresolved within 60d: {n_unresolved}")
    per_year = cars.groupby(cars["date"].dt.year).size()
    print(f"  per year: min {per_year.min()}, median {int(per_year.median())}, "
          f"max {per_year.max()} (2015: {per_year.get(2015, 0)})")
    print(f"  the sealed run you cannot get: mean {cars['missed'].mean():+.2%}, "
          f"median {cars['missed'].median():+.2%}, max {cars['missed'].max():+.1%}")
    print(f"  entry wait: 1d {(cars['wait'] == 1).mean():.1%}, "
          f">=2d {(cars['wait'] >= 2).mean():.1%}, >=3d {(cars['wait'] >= 3).mean():.1%}")

    rows = []
    for h in HORIZONS:
        rows.append(_report(f"{h}d gross", cars, f"car_{h}"))
        rows.append(_report(f"{h}d NET", cars, f"car_{h}_net"))
    _print(rows, "LIMIT-UP -- buy the first fillable close after the seal:")

    # --- pre-registered sub-samples at the verdict horizon --------------------------------
    col = f"car_{VERDICT_HORIZON}_net"
    pct = limit_pct_panel(close.index, close.columns)
    width = [float(pct.at[d, c]) for c, d in zip(cars["code"], cars["date"])]
    cars["width"] = width
    cars["board"] = [board_of(c) for c in cars["code"]]
    cars["hs300"] = [c in hs_sets[d] for c, d in zip(cars["code"], cars["date"])]
    rows = [
        _report("width 10%", cars[cars["width"] == 0.10], col),
        _report("width 20%", cars[cars["width"] == 0.20], col),
        _report("wait = 1d", cars[cars["wait"] == 1], col),
        _report("wait >= 2d", cars[cars["wait"] >= 2], col),
    ]
    cars["missed_tercile"] = pd.qcut(cars["missed"].rank(method="first"), 3,
                                     labels=["small", "mid", "large"])
    for g in ["small", "mid", "large"]:
        rows.append(_report(f"missed {g}", cars[cars["missed_tercile"] == g], col))
    for a, b in ERAS:
        rows.append(_report(f"{a[:4]}-{b[:4]}",
                            cars[(cars["date"] >= a) & (cars["date"] <= b)], col))
    rows.append(_report("HS300 member", cars[cars["hs300"]], col))
    rows.append(_report("CSI500-only", cars[~cars["hs300"]], col))
    _print(rows, f"{VERDICT_HORIZON}d NET -- pre-registered sub-samples:")

    # --- context leg: limit-DOWN (catching the falling knife) -----------------------------
    dn, dn_unresolved = build(direction=-1)
    _print([_report(f"{h}d NET", dn, f"car_{h}_net") for h in HORIZONS],
           f"LIMIT-DOWN context leg ({len(dn):,} events, {dn_unresolved} unresolved) -- "
           f"buy the first fillable close:")

    atomic_to_parquet(cars.drop(columns=["missed_tercile"]),
                      BACKTESTS_DIR / "limit_up_events.parquet")

    # --- the frozen verdict ----------------------------------------------------------------
    v = _report("verdict", cars, col)
    confirmed = v["mean"] > 0 and v["t_month"] > 2.0
    bleeds = v["mean"] < 0 and v["t_month"] < -2.0
    print(f"\nVERDICT (issue #1 frozen rule -- first fillable close, {VERDICT_HORIZON}d net, "
          f"monthly-clustered t): {'CONFIRMED' if confirmed else 'REJECTED'}")
    print(f"  mean abnormal {v['mean']:+.2%}, t(month) {v['t_month']:.2f}, "
          f"hit {v['hit']:.1%}, N {v['n']:,}")

    if not (confirmed or bleeds):
        return
    # Whichever direction cleared the bar is an AFFIRMATIVE claim; stress it.
    x = cars[col].dropna()
    d = cars.loc[x.index, "date"]
    tag = "CONTINUATION (the folklore wins here)" if confirmed else \
        "the A-share counterpart of the US post-gap bleed"
    print(f"\n  {tag}. Robustness:")
    for lo, hi, wtag in [(0.01, 0.99, "winsorized 1%"), (0.05, 0.95, "winsorized 5%")]:
        w = x.clip(x.quantile(lo), x.quantile(hi))
        print(f"    {wtag:>16}: mean {w.mean():+.2%}, t(month) "
              f"{clustered_tstat(w, d, freq='M'):.2f}")
    wins = int((x > 0).sum())
    p = stats.binomtest(wins, len(x), 0.5).pvalue
    print(f"    {'sign test':>16}: {wins:,}/{len(x):,} positive ({wins / len(x):.1%}), "
          f"binomial p = {p:.1e}")
    per_year = cars.groupby(cars["date"].dt.year)[col].mean()
    side = (per_year < 0) if bleeds else (per_year > 0)
    print(f"    {'per-year':>16}: {int(side.sum())}/{len(per_year)} years on the verdict's side")


if __name__ == "__main__":
    main()
