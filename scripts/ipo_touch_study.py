"""The multi-day touch ladder: rest a limit for a week, a month, a quarter? Issue #17.

Follow-up to #15/#16. The friend's proposal, repaired: rest a sell limit x% above the
day-1 open for N trading days; sell on touch, else at the day-N close.

  grid       N in {5, 20, 60} trading days; x in {2..50%}; complete windows only
  read       mean of T(N, x) FROM the day-1 open, t clustered by LISTING MONTH
  decision   21 cells -> the bar is t > 3 (one lucky t > 2 is likely by chance);
             otherwise the standing sell-at-the-open recommendation stands

    conda activate hermes
    python scripts/ipo_touch_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.io import atomic_to_parquet
from hermes.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from hermes.research.backtest.metrics import clustered_tstat

IPO_META_PARQUET = PARQUET_DIR / "ipo_star_meta.parquet"
IPO_BARS_PARQUET = PARQUET_DIR / "ipo_star_bars.parquet"
WINDOWS = (5, 20, 60)
LADDER = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
T_BAR = 3.0
ERAS = [("2019-01-01", "2021-12-31"), ("2022-01-01", "2026-12-31")]


def build_table() -> pd.DataFrame:
    meta = pd.read_parquet(IPO_META_PARQUET)
    bars = pd.read_parquet(IPO_BARS_PARQUET)
    rows = []
    for _, m in meta.iterrows():
        g = bars[bars["code"] == m["code"]].sort_values("date")
        if g.empty or not np.isfinite(g["open_raw"].iloc[0]) or g["open_raw"].iloc[0] <= 0:
            continue
        o = float(g["open_raw"].iloc[0])
        row = {"code": m["code"], "list_date": m["list_date"],
               "pop": o / m["issue_price"] - 1.0}
        for n in WINDOWS:
            if len(g) < n:
                for x in LADDER:
                    row[f"T_{n}_{x:.0%}"] = np.nan
                row[f"max_open_{n}"] = np.nan
                row[f"max_issue_{n}"] = np.nan
                continue
            hmax = float(g["high_raw"].iloc[:n].max())
            fallback = float(g["close_raw"].iloc[n - 1]) / o - 1.0
            row[f"max_open_{n}"] = hmax / o - 1.0
            row[f"max_issue_{n}"] = hmax / m["issue_price"] - 1.0
            for x in LADDER:
                row[f"T_{n}_{x:.0%}"] = x if hmax >= o * (1 + x) else fallback
        rows.append(row)
    return pd.DataFrame(rows)


def _grid(tag: str, t: pd.DataFrame) -> list[dict]:
    out = []
    for n in WINDOWS:
        r = {"tag": tag, "window": n}
        sub = t.dropna(subset=[f"T_{n}_{LADDER[0]:.0%}"])
        r["n"] = len(sub)
        for x in LADDER:
            s = sub[f"T_{n}_{x:.0%}"]
            r[f"L_{x:.0%}"] = float(s.mean()) if len(s) else np.nan
            r[f"t_{x:.0%}"] = (clustered_tstat(s, sub["list_date"], freq="M")
                               if len(s) > 2 else np.nan)
        out.append(r)
    return out


def _print(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    head = " ".join(f"{f'L({x:.0%})':>8} {'t':>6}" for x in LADDER)
    print(f"  {'sample':>22} {'N':>4} {'win':>4} {head}")
    for r in rows:
        cells = " ".join(f"{r[f'L_{x:.0%}']:>+8.2%} {r[f't_{x:.0%}']:>6.2f}" for x in LADDER)
        print(f"  {r['tag']:>22} {r['n']:>4} {r['window']:>3}d {cells}")


def main() -> None:
    ensure_dirs()
    t = build_table()
    print(f"cohort: {len(t)} STAR listings")

    print("\nwindow-max gain distributions (descriptive, no decision weight):")
    print(f"  {'window':>8} {'basis':>7} {'min':>8} {'p5':>8} {'p10':>8} {'p25':>8} {'med':>8}")
    for n in WINDOWS:
        for basis in ("open", "issue"):
            s = t[f"max_{basis}_{n}"].dropna()
            q = s.quantile([0.05, 0.10, 0.25, 0.50])
            print(f"  {n:>7}d {basis:>7} {s.min():>+8.1%} {q[0.05]:>+8.1%} "
                  f"{q[0.10]:>+8.1%} {q[0.25]:>+8.1%} {q[0.50]:>+8.1%}")

    rows = _grid("ALL (vs open = 0)", t)
    for s, e in ERAS:
        rows += _grid(f"{s[:4]}-{e[:4]}", t[(t["list_date"] >= s) & (t["list_date"] <= e)])
    _print(rows, "TOUCH LADDER RETURN FROM THE DAY-1 OPEN (day-N close if untouched):")

    subs = []
    for lab, grp in t.groupby(pd.qcut(t["pop"], 3, labels=["low", "mid", "high"]),
                              observed=True):
        subs += _grid(f"open pop {lab}", grp)
    broke = t[t["pop"] < 0]
    if len(broke):
        subs += _grid("broke issue", broke)
    _print(subs, "pre-registered sub-reads:")

    best, best_mean = None, 0.0
    for r in rows[:len(WINDOWS)]:
        for x in LADDER:
            if r[f"L_{x:.0%}"] > best_mean and r[f"t_{x:.0%}"] > T_BAR:
                best, best_mean = (r["window"], x), r[f"L_{x:.0%}"]
    if best is None:
        print(f"\nDECISION (frozen rule, t > {T_BAR:.0f} across a 21-cell grid): no cell "
              "qualifies -- the standing SELL AT THE DAY-1 OPEN remains.")
    else:
        print(f"\nDECISION (frozen rule): limit {best[1]:.0%} above the open for "
              f"{best[0]} days qualifies ({best_mean:+.2%}).")

    atomic_to_parquet(t, BACKTESTS_DIR / "ipo_touch_table.parquet", index=False)
    atomic_to_parquet(pd.DataFrame(rows + subs),
                      BACKTESTS_DIR / "ipo_touch_summary.parquet", index=False)
    print(f"saved -> {BACKTESTS_DIR / 'ipo_touch_summary.parquet'}")


if __name__ == "__main__":
    main()
