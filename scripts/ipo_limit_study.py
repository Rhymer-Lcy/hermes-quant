"""The limit ladder above the IPO open: is any gain level worth waiting for? Issue #16.

Follow-up to #15 (frozen decision: sell at the day-1 open). The executable version of
"sell once it is up x%": rest a limit at open x (1+x); if the day-1 high never touches it,
sell at the day-1 close.

  ladder     x in {2%, 5%, 10%, 15%, 20%, 30%}; fill at exactly the limit when the raw
             high touches it (optimistic by <= one tick, no queue risk -- disclosed)
  read       mean of L(x) measured FROM the open (sell cost cancels), t clustered by
             LISTING MONTH
  decision   recommend the best ladder level ONLY if its mean is POSITIVE with
             monthly-clustered t > 2; otherwise #15's sell-at-the-open stands

    conda activate hermes
    python scripts/ipo_limit_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.io import atomic_to_parquet
from hermes.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from hermes.research.backtest.metrics import clustered_tstat

IPO_META_PARQUET = PARQUET_DIR / "ipo_star_meta.parquet"
IPO_BARS_PARQUET = PARQUET_DIR / "ipo_star_bars.parquet"
LADDER = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
ERAS = [("2019-01-01", "2021-12-31"), ("2022-01-01", "2026-12-31")]


def build_table() -> pd.DataFrame:
    meta = pd.read_parquet(IPO_META_PARQUET)
    bars = pd.read_parquet(IPO_BARS_PARQUET)
    day1 = bars.sort_values("date").groupby("code").first().reset_index()
    t = day1.merge(meta[["code", "issue_price", "list_date"]], on="code", how="inner")
    t = t[(t["open_raw"] > 0) & t["high_raw"].notna() & t["close_raw"].notna()].copy()
    t["pop"] = t["open_raw"] / t["issue_price"] - 1.0
    t["high_over_open"] = t["high_raw"] / t["open_raw"] - 1.0
    close_ret = t["close_raw"] / t["open_raw"] - 1.0
    for x in LADDER:
        t[f"L_{x:.0%}"] = np.where(t["high_over_open"] >= x, x, close_ret)
    return t


def _line(tag: str, t: pd.DataFrame) -> dict:
    out = {"tag": tag, "n": len(t)}
    for x in LADDER:
        s = t[f"L_{x:.0%}"]
        out[f"L_{x:.0%}"] = float(s.mean())
        out[f"t_{x:.0%}"] = (clustered_tstat(s, t["list_date"], freq="M")
                             if len(s) > 2 else np.nan)
    return out


def _print(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    head = " ".join(f"{f'L({x:.0%})':>8} {'t':>6}" for x in LADDER)
    print(f"  {'sample':>22} {'N':>5} {head}")
    for r in rows:
        cells = " ".join(f"{r[f'L_{x:.0%}']:>+8.2%} {r[f't_{x:.0%}']:>6.2f}" for x in LADDER)
        print(f"  {r['tag']:>22} {r['n']:>5} {cells}")


def main() -> None:
    ensure_dirs()
    t = build_table()
    print(f"cohort: {len(t)} STAR listings, "
          f"{t['list_date'].min().date()} -> {t['list_date'].max().date()}")
    print(f"context (no decision weight): day-1 high over open mean "
          f"{t['high_over_open'].mean():+.2%}, median {t['high_over_open'].median():+.2%}; "
          f"close over open mean {(t['close_raw'] / t['open_raw'] - 1).mean():+.2%}")

    rows = [_line("ALL (vs sell at open = 0)", t)]
    for s, e in ERAS:
        rows.append(_line(f"{s[:4]}-{e[:4]}",
                          t[(t["list_date"] >= s) & (t["list_date"] <= e)]))
    _print(rows, "LIMIT LADDER RETURN FROM THE DAY-1 OPEN (close fallback if unfilled):")

    subs = []
    for lab, grp in t.groupby(pd.qcut(t["pop"], 3, labels=["low", "mid", "high"]),
                              observed=True):
        subs.append(_line(f"open pop {lab}", grp))
    broke = t[t["pop"] < 0]
    if len(broke):
        subs.append(_line("broke issue (open<issue)", broke))
    _print(subs, "pre-registered sub-reads:")

    r0 = rows[0]
    best, best_mean = None, 0.0
    for x in LADDER:
        if r0[f"L_{x:.0%}"] > best_mean and r0[f"t_{x:.0%}"] > 2:
            best, best_mean = x, r0[f"L_{x:.0%}"]
    if best is None:
        print("\nDECISION (frozen rule): no ladder level beats selling at the open at "
              "monthly-clustered t > 2 -- #15's SELL AT THE DAY-1 OPEN stands.")
    else:
        print(f"\nDECISION (frozen rule): a limit {best:.0%} above the open beats selling "
              f"at the open ({best_mean:+.2%}, t {r0[f't_{best:.0%}']:.2f}).")

    atomic_to_parquet(t, BACKTESTS_DIR / "ipo_limit_table.parquet", index=False)
    atomic_to_parquet(pd.DataFrame(rows + subs),
                      BACKTESTS_DIR / "ipo_limit_summary.parquet", index=False)
    print(f"saved -> {BACKTESTS_DIR / 'ipo_limit_summary.parquet'}")


if __name__ == "__main__":
    main()
