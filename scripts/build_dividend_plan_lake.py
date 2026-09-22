"""Build a CURRENT (code, ex_date, dps) cash-dividend table from exchange dividend plans.

Why this exists. The repo's `data/parquet/dividends.parquet` is a point-in-time ex-date lake built
2026-07-17, and it stops at ex-date 2026-07-31. Any FY2025 final dividend going ex after that date
is missing, which is not a rounding issue: 中国神华 reads 1.03 against 3.24 a year earlier,
建设银行 0.20 against 0.59, 粤高速A exactly 0.00. A yield screen on that lake silently rejects the
largest and most dependable payers for a data reason, so a long-horizon dividend study cannot use
it at the right-hand edge.

Eastmoney's `stock_fhps_em` publishes, per report period, every listed company's distribution plan
INCLUDING its 除权除息日. Summing the four quarterly snapshots of each fiscal year and keying on the
declared ex-date reconstructs the same (code, ex_date, dps) shape from a source that is current.

Two things this file is careful about, both learned the hard way here:

  * A fiscal year's distributions do NOT all go ex in the following calendar year -- for FY2024,
    886 plan rows went ex inside calendar 2024 and 3,832 in 2025. Fiscal-year totals and
    calendar-year ex-date totals are therefore different quantities and must never be compared
    as if interchangeable.
  * Many large payers now distribute twice (interim + final). A source that reports only the
    annual plan understates their payout by roughly half, which inverts payout-ratio conclusions
    for exactly the banks that dominate a dividend shortlist.

The build gates on amount-level agreement with the local lake over ex-dates BOTH sources cover.

    conda activate hermes
    python scripts/build_dividend_plan_lake.py
"""
from __future__ import annotations

import pandas as pd

from hermes.io import atomic_to_parquet
from hermes.paths import PARQUET_DIR, RESULTS_DIR, ensure_dirs

FYS = list(range(2015, 2027))
QUARTERS = ("0331", "0630", "0930", "1231")
OUT = RESULTS_DIR / "screens" / "dividend_plans.parquet"
MATCH_TOL = 0.005          # CNY per share; plans are quoted per 10 shares to 2-4 decimals


def pull() -> pd.DataFrame:
    import akshare as ak

    frames = []
    for fy in FYS:
        got = 0
        for q in QUARTERS:
            try:
                df = ak.stock_fhps_em(date=f"{fy}{q}")
            except Exception:                        # noqa: BLE001 -- a period may not exist yet
                continue
            if df is None or df.empty:
                continue
            cc = next(c for c in df.columns if c.strip() == "代码")
            pc = next(c for c in df.columns if c.strip() == "现金分红-现金分红比例")
            ec = next(c for c in df.columns if c.strip() == "除权除息日")
            x = pd.DataFrame({
                "k": df[cc].astype(str),
                "dps": pd.to_numeric(df[pc], errors="coerce").fillna(0.0) / 10.0,
                "ex_date": pd.to_datetime(df[ec], errors="coerce"),
                "fy": fy, "period": q,
            })
            x = x[(x["dps"] > 0) & x["ex_date"].notna()]
            frames.append(x)
            got += len(x)
        print(f"  FY{fy}: {got} cash distributions with a declared ex-date")
    out = pd.concat(frames, ignore_index=True)
    out["code"] = ["sh." + k if k.startswith(("6", "9")) else "sz." + k for k in out["k"]]
    out = (out.drop(columns=["k"])
              .drop_duplicates(subset=["code", "ex_date"], keep="last")
              .sort_values(["code", "ex_date"]).reset_index(drop=True))
    return out


def verify(plans: pd.DataFrame) -> None:
    """GATE: on ex-dates both sources cover, the AMOUNTS must agree."""
    lake = pd.read_parquet(PARQUET_DIR / "dividends.parquet")
    lake["ex_date"] = pd.to_datetime(lake["ex_date"])
    lake = lake[lake["dps"] > 0].groupby(["code", "ex_date"], as_index=False)["dps"].sum()
    m = plans.merge(lake, on=["code", "ex_date"], suffixes=("_plan", "_lake"))
    if m.empty:
        raise SystemExit("ABORT: no overlapping (code, ex_date) pairs to verify against")
    d = (m["dps_plan"] - m["dps_lake"]).abs()
    agree = float((d <= MATCH_TOL).mean())
    print(f"\n  GATE: {len(m)} (code, ex_date) pairs present in BOTH sources; "
          f"{agree:.2%} agree to within CNY {MATCH_TOL}/share")
    print(f"        median |diff| {d.median():.6f}   p99 {d.quantile(0.99):.6f}   "
          f"max {d.max():.6f}")
    if agree < 0.97:
        worst = m.assign(d=d).nlargest(8, "d")
        print(worst[["code", "ex_date", "dps_plan", "dps_lake", "d"]].to_string(index=False))
        raise SystemExit("ABORT: plan amounts disagree with the ex-date lake")
    print("        -> PASS")

    # Coverage at the right-hand edge is the whole point of this rebuild.
    cut = pd.Timestamp("2026-07-31")
    print(f"\n  coverage past the old lake's last ex-date ({cut.date()}): "
          f"{int((plans['ex_date'] > cut).sum())} distributions the lake does not have, "
          f"{plans.loc[plans['ex_date'] > cut, 'code'].nunique()} distinct names")


def main() -> None:
    ensure_dirs()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plans = pull()
    print(f"\ntotal: {len(plans)} distributions, {plans['code'].nunique()} names, "
          f"{plans['ex_date'].min().date()} -> {plans['ex_date'].max().date()}")
    verify(plans)
    atomic_to_parquet(plans, OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
