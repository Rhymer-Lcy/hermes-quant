"""The ROE-slope anchor: buy far below the compounding line. Issue #10.

The friend's theory, pre-registered before this script existed: a stable compounder's
total-return price tracks a line compounding at its average ROE; buying when the price sits
far below that line earns excess returns.

  universe   PIT HS300+CSI500 members, non-ST, >= 20 prior traded days, restricted PIT to
             stable profitable names (every published annual ROE > 0, mean >= 10%)
  anchor     per name from t0 = first traded day after the 3rd published annual report:
             anchor(t) = adj_close(t0) * (1 + mean published ROE known at t)^(years since t0)
  signal     deviation = adj_close/anchor - 1 <= -30%
  verdict    CONFIRMED only if the monthly EW deviation portfolio beats the EW universe net
             with monthly-clustered t > 2

    conda activate hermes
    python scripts/roe_anchor_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.data.fundamentals import load_annual_roe
from hermes.data.lake import load_close_panel
from hermes.data.membership import (CSI500_MEMBERSHIP_PARQUET, MEMBERSHIP_PARQUET,
                                    membership_lookup)
from hermes.io import atomic_to_parquet
from hermes.paths import BACKTESTS_DIR, ensure_dirs
from hermes.research.backtest.metrics import clustered_tstat, sharpe
from hermes.research.backtest.rule_portfolio import month_end_dates, monthly_ew_backtest

DEV_BUY = -0.30
MIN_MEAN_ROE = 0.10
BASE_REPORT, REBASE_REPORT = 3, 5
EXAMPLE = "sh.600900"          # the friend's hydropower illustration
ERAS = [("2015-01-01", "2019-12-31"), ("2020-01-01", "2026-12-31")]


def anchor_panels(close: pd.DataFrame, base_report: int = BASE_REPORT
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(deviation, stable): per-name deviation from the PIT ROE-slope anchor, and the PIT
    stable-profitable mask (>= `base_report` published reports, all ROE > 0, mean >= 10%)."""
    roe = load_annual_roe()
    dev = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    stable = pd.DataFrame(False, index=close.index, columns=close.columns)
    for code, grp in roe.groupby("code"):
        if code not in close.columns:
            continue
        g = grp.sort_values("pubDate")
        pubs = g["pubDate"].to_numpy()
        vals = g["roeAvg"].to_numpy(dtype=float)
        mean = pd.Series(np.cumsum(vals) / np.arange(1, len(vals) + 1), index=pubs)
        allpos = pd.Series(np.minimum.accumulate(vals) > 0, index=pubs)
        px = close[code].dropna()
        if len(pubs) < base_report or px.empty:
            continue
        t0_pos = px.index.searchsorted(pubs[base_report - 1], side="right")
        if t0_pos >= len(px):
            continue
        t0, p0 = px.index[t0_pos], float(px.iloc[t0_pos])
        mean_t = mean[~mean.index.duplicated(keep="last")].reindex(close.index, method="ffill")
        ok_t = allpos[~allpos.index.duplicated(keep="last")].reindex(
            close.index, method="ffill").eq(True)
        years = (close.index - t0).days / 365.25
        anchor = p0 * (1.0 + mean_t) ** years
        d = close[code] / anchor - 1.0
        live = close.index >= t0
        dev.loc[live, code] = d[live]
        stable[code] = live & ok_t.to_numpy() & (mean_t >= MIN_MEAN_ROE).to_numpy()
    return dev, stable


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
    print(f"  {'variant':>26} {'wealth':>8} {'bench':>8} {'active/yr':>10} {'Sharpe':>7} "
          f"{'bench':>7} {'t(month)':>9}")
    for r in rows:
        print(f"  {r['tag']:>26} {r['wealth_strat']:>8.3f} {r['wealth_bench']:>8.3f} "
              f"{r['ann_active']:>+10.2%} {r['sharpe_strat']:>7.2f} {r['sharpe_bench']:>7.2f} "
              f"{r['t_month']:>9.2f}")


def main() -> None:
    ensure_dirs()
    hs = pd.read_parquet(MEMBERSHIP_PARQUET)
    cs = pd.read_parquet(CSI500_MEMBERSHIP_PARQUET)
    union = sorted(set(hs["code"]) | set(cs["code"]))
    close = load_close_panel(codes=union, field="close")
    st = load_close_panel(codes=union, field="isST")
    hs_asof, cs_asof = membership_lookup(hs), membership_lookup(cs)
    member = pd.DataFrame({c: [c in hs_asof(d) or c in cs_asof(d) for d in close.index]
                           for c in close.columns}, index=close.index)
    base = (member & ~st.eq(True) & close.notna()
            & (close.notna().cumsum().shift(1) >= 20))
    ret = close.pct_change(fill_method=None)

    dev, stable = anchor_panels(close)
    pool = base & stable
    print(f"stable-profitable pool: {pool.sum(axis=1).mean():.0f} names/day of "
          f"{base.sum(axis=1).mean():.0f}; deviation <= {DEV_BUY:.0%}: "
          f"{(pool & (dev <= DEV_BUY)).sum(axis=1).mean():.0f} names/day")

    bh = monthly_ew_backtest(base, ret)["net"]
    buy = monthly_ew_backtest(pool & (dev <= DEV_BUY), ret)["net"]
    rows = [_line("dev <= -30% vs universe NET", buy, bh)]
    for s, e in ERAS:
        sl = slice(pd.Timestamp(s), pd.Timestamp(e))
        rows.append(_line(f"{s[:4]}-{e[:4]}", buy.loc[sl], bh.loc[sl]))
    for th in (-0.20, -0.40):
        rows.append(_line(f"threshold {th:.0%}",
                          monthly_ew_backtest(pool & (dev <= th), ret)["net"], bh))
    dev5, stable5 = anchor_panels(close, base_report=REBASE_REPORT)
    rows.append(_line("re-based at 5th report",
                      monthly_ew_backtest(base & stable5 & (dev5 <= DEV_BUY), ret)["net"],
                      bh))
    _print(rows, "ROE-SLOPE ANCHOR (buy far below the compounding line):")

    # Frozen shape read: forward 250d abnormal CAR by deviation quintile.
    abn = ret.sub(ret.where(base).mean(axis=1), axis=0)
    fwd = abn[::-1].rolling(250, min_periods=250).sum()[::-1].shift(-1)
    recs = []
    for d in month_end_dates(close.index):
        x = dev.loc[d].where(pool.loc[d])
        f = fwd.loc[d]
        both = pd.DataFrame({"dev": x, "fwd": f}).dropna()
        if len(both) < 50:
            continue
        both["q"] = pd.qcut(both["dev"], 5, labels=False, duplicates="drop")
        for q, grp in both.groupby("q"):
            recs.append({"date": d, "q": int(q), "fwd": float(grp["fwd"].mean())})
    qdf = pd.DataFrame(recs)
    print("\nforward 250d abnormal by deviation quintile (0 = deepest below anchor):")
    for q, grp in qdf.groupby("q"):
        print(f"  Q{q}: {grp['fwd'].mean():+7.2%}  "
              f"(t_month {clustered_tstat(grp['fwd'], grp['date'], freq='M'):.2f})")

    # Frozen descriptive read: mean published ROE vs realized total-return CAGR.
    roe = load_annual_roe()
    pairs = []
    for code, grp in roe.groupby("code"):
        if code not in close.columns or len(grp) < 8:
            continue
        px = close[code].dropna()
        if len(px) < 8 * 250:
            continue
        yrs = (px.index[-1] - px.index[0]).days / 365.25
        pairs.append({"code": code, "mean_roe": float(grp["roeAvg"].mean()),
                      "cagr": float((px.iloc[-1] / px.iloc[0]) ** (1 / yrs) - 1)})
    pdf = pd.DataFrame(pairs)
    corr = pdf["mean_roe"].corr(pdf["cagr"])
    print(f"\ndescriptive: corr(mean ROE, realized CAGR) = {corr:.2f} over {len(pdf)} names "
          f"with >= 8 years of both")
    ex = pdf[pdf["code"] == EXAMPLE]
    if not ex.empty:
        print(f"  the utility example: mean ROE {ex['mean_roe'].iloc[0]:.1%}, "
              f"realized CAGR {ex['cagr'].iloc[0]:.1%}")
    slope = float(np.polyfit(pdf["mean_roe"], pdf["cagr"], 1)[0])
    t_corr = corr * np.sqrt((len(pdf) - 2) / max(1e-12, 1 - corr ** 2))
    print(f"  cross-sectional slope: {slope:.2f} (the friend's claim implies ~1), "
          f"t of corr {t_corr:.2f}")

    head = rows[0]
    verdict = head["ann_active"] > 0 and head["t_month"] > 2
    print(f"\nVERDICT: {'CONFIRMED' if verdict else 'REJECTED'} -- needs dev <= -30% "
          f"portfolio > EW universe net with monthly-clustered t > 2 (got {head['t_month']:.2f})")

    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "roe_anchor_summary.parquet",
                      index=False)
    atomic_to_parquet(qdf, BACKTESTS_DIR / "roe_anchor_quintiles.parquet", index=False)
    print(f"saved -> {BACKTESTS_DIR / 'roe_anchor_summary.parquet'}")


if __name__ == "__main__":
    main()
