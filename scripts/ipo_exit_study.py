"""The allottee's exit: when to sell a STAR Market IPO allocation. Issue #15.

Pre-registered decision study (no friend's rule on trial): across every STAR listing
since the board opened, which frozen exit rule maximized the allottee's net return?

  exits      day-1 open, then closes of trading days 1/5/20/60/120
  return     (day-1 raw open / issue) x adjusted total-return factor to the exit, net of
             the proportional sell-side retail cost (12.6bp; the Y5 commission minimum is
             excluded per series convention)
  pairing    per IPO, holding return from the day-1 open to exit X (the sell cost cancels)
  decision   recommend holding to X over selling the open ONLY if the paired mean is
             POSITIVE with listing-month-clustered t > 2; otherwise default to the open

    conda activate hermes
    python scripts/ipo_exit_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.io import atomic_to_parquet
from hermes.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from hermes.research.backtest.metrics import clustered_tstat

IPO_META_PARQUET = PARQUET_DIR / "ipo_star_meta.parquet"
IPO_BARS_PARQUET = PARQUET_DIR / "ipo_star_bars.parquet"
SELL_COST = 0.00126             # commission 2.5bp + stamp 5bp + transfer 0.1bp + slip 5bp
EXIT_DAYS = (1, 5, 20, 60, 120)   # closes of these trading days (day 1 = listing day)
MEGA_PROCEEDS = 1e10
ERAS = [("2019-01-01", "2021-12-31"), ("2022-01-01", "2026-12-31")]


def build_table() -> pd.DataFrame:
    meta = pd.read_parquet(IPO_META_PARQUET)
    bars = pd.read_parquet(IPO_BARS_PARQUET)
    rows = []
    for _, m in meta.iterrows():
        g = bars[bars["code"] == m["code"]].sort_values("date")
        if g.empty or not np.isfinite(g["open_raw"].iloc[0]):
            continue
        o_raw, o_hfq = float(g["open_raw"].iloc[0]), float(g["open_hfq"].iloc[0])
        if o_raw <= 0 or o_hfq <= 0:
            continue
        row = {"code": m["code"], "name": m["name"], "list_date": m["list_date"],
               "issue": float(m["issue_price"]), "proceeds": float(m["proceeds"]),
               "pop": o_raw / m["issue_price"] - 1.0,
               "ret_open": (o_raw / m["issue_price"]) * (1 - SELL_COST) - 1.0}
        for d in EXIT_DAYS:
            if len(g) >= d:
                f = float(g["close_hfq"].iloc[d - 1]) / o_hfq
                row[f"hold_{d}"] = f - 1.0
                row[f"ret_{d}"] = (o_raw / m["issue_price"]) * f * (1 - SELL_COST) - 1.0
            else:
                row[f"hold_{d}"] = np.nan
                row[f"ret_{d}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _hold_line(tag: str, t: pd.DataFrame) -> dict:
    out = {"tag": tag, "n": len(t)}
    for d in EXIT_DAYS:
        x = t[f"hold_{d}"].dropna()
        out[f"hold_{d}"] = float(x.mean()) if len(x) else np.nan
        out[f"t_{d}"] = (clustered_tstat(x, t.loc[x.index, "list_date"], freq="M")
                         if len(x) > 2 else np.nan)
    return out


def _print_holds(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    head = " ".join(f"{f'd{d} close':>10} {'t(mo)':>6}" for d in EXIT_DAYS)
    print(f"  {'sample':>22} {'N':>5} {head}")
    for r in rows:
        cells = " ".join(f"{r[f'hold_{d}']:>+10.2%} {r[f't_{d}']:>6.2f}" for d in EXIT_DAYS)
        print(f"  {r['tag']:>22} {r['n']:>5} {cells}")


def main() -> None:
    ensure_dirs()
    t = build_table()
    print(f"cohort: {len(t)} STAR listings, "
          f"{t['list_date'].min().date()} -> {t['list_date'].max().date()}")

    print("\nallottee NET return vs issue price by exit (cohort mean / median):")
    print(f"  {'day-1 open':>12} {t['ret_open'].mean():>+9.1%} {t['ret_open'].median():>+9.1%}")
    for d in EXIT_DAYS:
        x = t[f"ret_{d}"].dropna()
        print(f"  {f'd{d} close':>12} {x.mean():>+9.1%} {x.median():>+9.1%}")

    rows = [_hold_line("ALL (vs day-1 open)", t)]
    for s, e in ERAS:
        rows.append(_hold_line(f"{s[:4]}-{e[:4]}",
                               t[(t["list_date"] >= s) & (t["list_date"] <= e)]))
    _print_holds(rows, "HOLDING RETURN FROM THE DAY-1 OPEN (the decision quantity):")

    subs = []
    for lab, grp in t.groupby(pd.qcut(t["proceeds"], 3, labels=["small", "mid", "large"]),
                              observed=True):
        subs.append(_hold_line(f"proceeds {lab}", grp))
    for lab, grp in t.groupby(pd.qcut(t["pop"], 3, labels=["low", "mid", "high"]),
                              observed=True):
        subs.append(_hold_line(f"open pop {lab}", grp))
    broke = t[t["pop"] < 0]
    if len(broke):
        subs.append(_hold_line("broke issue (open<issue)", broke))
    _print_holds(subs, "pre-registered sub-reads:")

    mega = t[t["proceeds"] >= MEGA_PROCEEDS].sort_values("list_date")
    print(f"\nmega cohort (proceeds >= CNY {MEGA_PROCEEDS:.0e}), descriptive:")
    for _, r in mega.iterrows():
        d20 = "open" if pd.isna(r["ret_20"]) else f"{r['ret_20']:+.0%}"
        d120 = "open" if pd.isna(r["ret_120"]) else f"{r['ret_120']:+.0%}"
        print(f"  {r['list_date'].date()} {r['name']:>6} issue {r['issue']:>7.2f} "
              f"pop {r['pop']:>+7.1%} d20 {d20:>6} d120 {d120:>6}")

    best = None
    for d in EXIT_DAYS:
        r0 = rows[0]
        if r0[f"hold_{d}"] > 0 and r0[f"t_{d}"] > 2:
            best = d if best is None or rows[0][f"hold_{d}"] > rows[0][f"hold_{best}"] else best
    if best is None:
        print("\nDECISION (frozen rule): no exit beats the day-1 open at monthly-clustered "
              "t > 2 -- the recommendation defaults to SELLING AT THE DAY-1 OPEN.")
    else:
        print(f"\nDECISION (frozen rule): holding to the day-{best} close beats the open "
              f"({rows[0][f'hold_{best}']:+.2%}, t {rows[0][f't_{best}']:.2f}) -- "
              f"the recommendation is to hold to day {best}.")

    atomic_to_parquet(t, BACKTESTS_DIR / "ipo_exit_table.parquet", index=False)
    atomic_to_parquet(pd.DataFrame(rows + subs),
                      BACKTESTS_DIR / "ipo_exit_summary.parquet", index=False)
    print(f"saved -> {BACKTESTS_DIR / 'ipo_exit_summary.parquet'}")


if __name__ == "__main__":
    main()
