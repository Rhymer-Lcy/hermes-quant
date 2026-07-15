"""Quality-value entry: three years of ROE > 15%, PE under its own 30th percentile. Issue #2.

The friend's core buy rule, pre-registered before this script existed:

  universe   PIT HS300+CSI500 members, non-ST, >= 20 prior traded days, 2015 -> present
  quality    the three most recently PUBLISHED annual reports each show ROE > 15%
             (publication-date aligned -- a report counts only from its pubDate)
  value      peTTM > 0 and at or below the name's own trailing 1200d 30th percentile
             (>= 750 observations required)
  arms       P_qv (quality AND value), P_q (quality only -- what the PE gate adds),
             benchmark = equal-weight universe
  portfolio  equal weight, monthly rebalance, signal month-end t -> execute t+1 close,
             proportional retail costs
  verdict    CONFIRMED only if P_qv beats the EW universe net with monthly-clustered t > 2;
             P_qv minus P_q is the frozen attribution read (reported, not the verdict)

Also computes and saves the issue #3 event set (days a name NEWLY satisfies both gates).

    conda activate hermes
    python scripts/quality_value_study.py
"""
from __future__ import annotations

import pandas as pd

from hermes.data.fundamentals import INDUSTRY_PARQUET, quality_mask
from hermes.data.lake import load_close_panel
from hermes.data.membership import (CSI500_MEMBERSHIP_PARQUET, MEMBERSHIP_PARQUET,
                                    membership_lookup)
from hermes.io import atomic_to_parquet
from hermes.paths import BACKTESTS_DIR, ensure_dirs
from hermes.research.backtest.metrics import clustered_tstat, sharpe
from hermes.research.backtest.rule_portfolio import (monthly_ew_backtest,
                                                     rolling_own_quantile)

PCT_VALUE = 0.30
WINDOW, MIN_OBS = 1200, 750
MIN_HISTORY = 20
ERAS = [("2015-01-01", "2019-12-31"), ("2020-01-01", "2026-12-31")]


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
    print(f"  {'variant':>30} {'wealth':>8} {'bench':>8} {'active/yr':>10} {'Sharpe':>7} "
          f"{'bench':>7} {'t(month)':>9}")
    for r in rows:
        print(f"  {r['tag']:>30} {r['wealth_strat']:>8.3f} {r['wealth_bench']:>8.3f} "
              f"{r['ann_active']:>+10.2%} {r['sharpe_strat']:>7.2f} {r['sharpe_bench']:>7.2f} "
              f"{r['t_month']:>9.2f}")


def main() -> None:
    ensure_dirs()
    hs = pd.read_parquet(MEMBERSHIP_PARQUET)
    cs = pd.read_parquet(CSI500_MEMBERSHIP_PARQUET)
    union = sorted(set(hs["code"]) | set(cs["code"]))
    close = load_close_panel(codes=union, field="close")
    pe = load_close_panel(codes=union, field="peTTM")
    st = load_close_panel(codes=union, field="isST")
    hs_asof, cs_asof = membership_lookup(hs), membership_lookup(cs)
    member = pd.DataFrame({c: [c in hs_asof(d) or c in cs_asof(d) for d in close.index]
                           for c in close.columns}, index=close.index)
    base = (member & ~st.eq(True) & close.notna()
            & (close.notna().cumsum().shift(1) >= MIN_HISTORY))
    ret = close.pct_change(fill_method=None)

    quality = quality_mask(close.index, close.columns)
    q30 = rolling_own_quantile(pe, PCT_VALUE, WINDOW, MIN_OBS)
    value = (pe > 0) & (pe <= q30)
    masks = {"P_qv": base & quality & value, "P_q": base & quality, "universe": base}
    print(f"universe: {len(union)} names; quality names/day "
          f"{(base & quality).sum(axis=1).mean():.0f}, quality+value "
          f"{masks['P_qv'].sum(axis=1).mean():.0f}, universe {base.sum(axis=1).mean():.0f}")

    res = {k: monthly_ew_backtest(m, ret) for k, m in masks.items()}
    qv, q, uni = res["P_qv"]["net"], res["P_q"]["net"], res["universe"]["net"]

    rows = [_line("P_qv vs universe NET", qv, uni),
            _line("P_q vs universe NET", q, uni),
            _line("P_qv vs P_q (the PE gate)", qv, q)]
    for s, e in ERAS:
        sl = slice(pd.Timestamp(s), pd.Timestamp(e))
        rows.append(_line(f"P_qv vs universe {s[:4]}-{e[:4]}", qv.loc[sl], uni.loc[sl]))

    # Frozen sub-samples: HS300 members vs CSI500-only, at the qv mask level.
    hs_member = pd.DataFrame({c: [c in hs_asof(d) for d in close.index]
                              for c in close.columns}, index=close.index)
    for tag, cut in [("HS300 members", hs_member), ("CSI500-only", ~hs_member)]:
        sub_qv = monthly_ew_backtest(masks["P_qv"] & cut, ret)["net"]
        sub_uni = monthly_ew_backtest(base & cut, ret)["net"]
        rows.append(_line(f"{tag} qv vs universe", sub_qv, sub_uni))

    # Frozen sub-samples: the friend's five-sector subset, and the 20%-per-bucket variant.
    buckets = pd.read_parquet(INDUSTRY_PARQUET).set_index("code")["bucket"]
    in_five = pd.Series({c: pd.notna(buckets.get(c)) for c in close.columns})
    five_cols = list(in_five.index[in_five])
    rows.append(_line("five-sector qv vs universe",
                      monthly_ew_backtest(masks["P_qv"][five_cols], ret[five_cols])["net"], uni))
    rows.append(_line("off-map qv vs universe",
                      monthly_ew_backtest(masks["P_qv"].drop(columns=five_cols),
                                          ret.drop(columns=five_cols))["net"], uni))
    sleeves = []
    for b in ("finance", "livelihood", "consumption", "tech", "infrastructure"):
        cols = [c for c in close.columns if buckets.get(c) == b]
        sleeves.append(monthly_ew_backtest(masks["P_qv"][cols], ret[cols])["net"])
    balanced = sum(s * 0.2 for s in sleeves)
    rows.append(_line("20%-per-bucket qv vs universe", balanced, uni))
    rows.append(_line("20%-per-bucket vs plain qv", balanced, qv))

    _print(rows, "QUALITY-VALUE ENTRY (3yr ROE > 15%, PE under own 30th pct):")

    head = rows[0]
    verdict = head["ann_active"] > 0 and head["t_month"] > 2
    print(f"\nVERDICT: {'CONFIRMED' if verdict else 'REJECTED'} -- needs P_qv > EW universe "
          f"net with monthly-clustered t > 2 (got {head['t_month']:.2f})")

    # The issue #3 event set: days a name NEWLY satisfies both gates.
    fresh = masks["P_qv"] & ~masks["P_qv"].shift(1, fill_value=False)
    events = fresh.stack()
    events = events[events]
    ev = pd.DataFrame({"date": events.index.get_level_values(0),
                       "code": events.index.get_level_values(1)}).sort_values(
                           ["date", "code"]).reset_index(drop=True)
    print(f"issue #3 event set: {len(ev)} fresh quality-value entries in "
          f"{ev['code'].nunique()} names")

    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "quality_value_summary.parquet",
                      index=False)
    atomic_to_parquet(pd.DataFrame({"qv_net": qv, "q_net": q, "universe_net": uni}),
                      BACKTESTS_DIR / "quality_value_daily.parquet")
    atomic_to_parquet(ev, BACKTESTS_DIR / "quality_value_events.parquet", index=False)
    print(f"saved -> {BACKTESTS_DIR / 'quality_value_summary.parquet'}")


if __name__ == "__main__":
    main()
