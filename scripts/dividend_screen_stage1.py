"""Stage 1 of a long-horizon A-share dividend screen: narrow the PIT HS300+CSI500 union
to a manageable candidate set using only local lake data plus today's quoted price.

This stage deliberately screens for DURABILITY, not for headline yield. Yield enters only as a
floor and is never the ranking key -- a collapsing share price raises the printed yield while
destroying the thesis, which is the single most common dividend trap.

Stage 2 (`dividend_screen_stage2.py`) then pulls full statements for the survivors and checks
cash-flow coverage, leverage and sector-appropriate quality, because those cannot be read off
this lake.

Outputs results/screens/dividend_stage1.parquet (gitignored). Read-only w.r.t. everything else.

    conda activate hermes
    python scripts/dividend_screen_stage1.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.data.fundamentals import INDUSTRY_PARQUET
from hermes.data.membership import CSI500_MEMBERSHIP_PARQUET, MEMBERSHIP_PARQUET
from hermes.io import atomic_to_parquet
from hermes.paths import PARQUET_DIR, RESULTS_DIR, ensure_dirs

TODAY = pd.Timestamp("2026-09-21")
OUT = RESULTS_DIR / "screens"
CYCLE_YEARS = list(range(2017, 2027))       # ex-date calendar years 2017..2026
RECENT = list(range(2021, 2027))            # the six most recent ex-date years
PLAN_LAKE = RESULTS_DIR / "screens" / "dividend_plans.parquet"
LATEST_FY = 2025


def dividend_tables():
    """(per-calendar-year DPS by ex-date, trailing-12-month DPS) from the plan lake.

    Built by `build_dividend_plan_lake.py`, which gates plan amounts against the local ex-date
    lake (99.85% agree, median difference exactly 0). Two disciplines are enforced here:

      * a fiscal year's distributions do NOT all go ex in one calendar year, so everything is
        bucketed by the DECLARED EX-DATE and never by fiscal year;
      * plans exist with ex-dates in the FUTURE (the table runs to 2026-10-23). Anything on or
        after TODAY is dropped -- a screen that counted an announced-but-not-yet-ex dividend
        would be using information the holder has not received.
    """
    plans = pd.read_parquet(PLAN_LAKE)
    plans["ex_date"] = pd.to_datetime(plans["ex_date"])
    future = int((plans["ex_date"] > TODAY).sum())
    plans = plans[plans["ex_date"] <= TODAY]
    print(f"  plan lake: {len(plans)} distributions on or before {TODAY.date()} "
          f"({future} future ex-dates dropped)")
    yr = plans.assign(y=plans["ex_date"].dt.year).groupby(["code", "y"])["dps"].sum().unstack(
        fill_value=0.0)
    ttm = (plans[plans["ex_date"] > TODAY - pd.Timedelta(days=365)]
           .groupby("code")["dps"].sum())
    # THE YIELD BASIS IS THE DECLARED FISCAL-YEAR PAYOUT, NOT TTM.
    #
    # A trailing-12-month window slices across fiscal years and produces artifacts whenever a
    # company changes its distribution cadence. Measured here: 成都银行's TTM catches the FY2025
    # FINAL (ex 2026-07-13, 0.9210) together with a brand-new FY2026 INTERIM (ex 2026-09-18,
    # 0.4890) and prints 7.50%, while its actual FY2025 payout is 0.9210 = 4.90%. Telling a
    # five-year holder the yield is 7.5% because two fiscal years happened to overlap one window
    # is precisely the trap this study exists to avoid.
    #
    # FY2025 is the most recent complete declared cycle. Amounts are counted as DECLARED (a plan
    # is public once announced) rather than as already-ex, so a company whose final has been
    # announced but not yet gone ex is not understated; `fy_not_yet_ex` records how much of it
    # the holder has not physically received.
    fy = LATEST_FY
    full = pd.read_parquet(PLAN_LAKE)
    full["ex_date"] = pd.to_datetime(full["ex_date"])
    declared = full[full["fy"] == fy].groupby("code")["dps"].sum()
    pending = (full[(full["fy"] == fy) & (full["ex_date"] > TODAY)]
               .groupby("code")["dps"].sum())
    print(f"  FY{fy} declared payers: {int((declared > 0).sum())}; "
          f"{int((pending > 0).sum())} still have an un-ex portion")
    return yr, declared, ttm, pending


def spot_prices() -> pd.Series:
    """Today's quoted (unadjusted) A-share prices, keyed to the lake's sh./sz. convention."""
    import akshare as ak

    df = ak.stock_zh_a_spot_em()
    code_col = next(c for c in df.columns if c.strip() == "代码")
    px_col = next(c for c in df.columns if c.strip() == "最新价")
    name_col = next(c for c in df.columns if c.strip() == "名称")
    out, names = {}, {}
    for c, p, nm in zip(df[code_col].astype(str), pd.to_numeric(df[px_col], errors="coerce"),
                        df[name_col].astype(str)):
        pre = "sh" if c.startswith(("6", "9")) else ("sz" if c.startswith(("0", "3", "2")) else None)
        if pre and np.isfinite(p) and p > 0:
            out[f"{pre}.{c}"] = float(p)
            names[f"{pre}.{c}"] = nm
    return pd.Series(out, name="price"), pd.Series(names, name="name")


def main() -> None:
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)

    hs = pd.read_parquet(MEMBERSHIP_PARQUET)
    cs = pd.read_parquet(CSI500_MEMBERSHIP_PARQUET)
    universe = sorted(set(hs["code"]) | set(cs["code"]))
    print(f"universe (PIT HS300 + CSI500 union): {len(universe)} names")

    # DPS comes from the exchange DIVIDEND PLANS, not from the local ex-date lake.
    #
    # The lake was built 2026-07-17 and stops at ex-date 2026-07-31, so for any company whose
    # FY2025 FINAL dividend went ex after that date the calendar-2026 sum is only a fraction of
    # the real payout -- measured: 中国神华 1.03 against 3.24 a year earlier, 建设银行 0.20
    # against 0.59, 粤高速A 0.00. Those names were then silently rejected by the yield floor,
    # i.e. the screen was discarding the largest and most reliable payers for a data reason.
    #
    # Summing the four quarterly plan snapshots of a fiscal year recovers the full payout
    # (interim + final) from a source that is current for every listed name. Verified against the
    # lake wherever the lake IS complete: 招商银行 2.0160, 长江电力 1.0000, 青岛啤酒 2.3500,
    # 中国建筑 0.2718 -- exact matches; and 中国移动 4.7037 independently reproduces the FY2025
    # DPS used in an earlier hand analysis.
    dps, declared, ttm, pending = dividend_tables()
    dps = dps.reindex(index=[c for c in universe if c in dps.index], columns=CYCLE_YEARS,
                      fill_value=0.0)

    roe = pd.read_parquet(PARQUET_DIR / "profit_annual.parquet")
    roe = roe[roe["code"].isin(universe)].copy()
    roe["fy"] = pd.to_datetime(roe["statDate"]).dt.year
    roe_w = roe.pivot_table(index="code", columns="fy", values="roeAvg", aggfunc="last")
    roe5 = roe_w.reindex(columns=range(2021, 2026))

    ind = pd.read_parquet(INDUSTRY_PARQUET).set_index("code")
    px, names = spot_prices()
    print(f"spot quotes: {len(px)} tickers")

    rows = []
    for code in universe:
        if code not in dps.index or code not in px.index:
            continue
        d = dps.loc[code]
        recent = d[RECENT]
        last = float(declared.get(code, 0.0))          # declared FY2025 payout -- the run-rate
        if last <= 0:
            continue
        ttm_v = float(ttm.get(code, 0.0))
        pend = float(pending.get(code, 0.0))
        paid10 = int((d > 0).sum())
        paid6 = int((recent > 0).sum())
        # longest run of consecutive paying ex-date years ending at the current one
        streak = 0
        for y in range(2026, min(CYCLE_YEARS) - 1, -1):
            if float(d[y]) > 0:
                streak += 1
            else:
                break
        base5 = float(d[2021])
        cagr5 = (float(d[2026]) / base5) ** (1 / 5) - 1 if base5 > 0 else np.nan
        cuts = int(sum(1 for a, b in zip(recent[:-1], recent[1:]) if b < a * 0.999))
        r = roe5.loc[code] if code in roe5.index else pd.Series(dtype=float)
        rows.append({
            "code": code, "name": names.get(code, ""),
            "csrc": ind["csrc"].get(code), "bucket": ind["bucket"].get(code),
            "price": float(px[code]), "dps_latest": last,
            "yield": last / float(px[code]),
            "dps_ttm": ttm_v, "yield_ttm": ttm_v / float(px[code]),
            "fy_not_yet_ex": pend,
            "ttm_vs_fy": (ttm_v / last) if last > 0 else float("nan"),
            "paid_10y": paid10, "paid_6y": paid6, "streak": streak,
            "dps_cagr5": cagr5, "dps_cuts_6y": cuts,
            "dps_2021": base5, "dps_2026": float(d[2026]),
            "roe_mean5": float(r.mean()) if len(r.dropna()) else np.nan,
            "roe_min5": float(r.min()) if len(r.dropna()) else np.nan,
            "roe_std5": float(r.std()) if len(r.dropna()) > 1 else np.nan,
            "roe_n": int(r.notna().sum()),
        })
    df = pd.DataFrame(rows).set_index("code")
    print(f"\nnames with a FY2025-cycle payout and a live quote: {len(df)}")

    # ---- durability filters. Yield is a FLOOR, never the ranking key. ----
    f = df[(df["paid_6y"] >= 6)                      # paid in every one of the last six cycles
           & (df["streak"] >= 6)
           & (df["roe_n"] >= 4)
           & (df["roe_min5"] > 0.0)                  # no loss year in five
           & (df["yield"] >= 0.025)]
    print(f"after durability filters (6/6 cycles, unbroken streak, no loss year, yield >= 2.5%): "
          f"{len(f)}")
    f = f[(f["dps_cuts_6y"] <= 2)]
    print(f"after 'at most two DPS cuts in six cycles': {len(f)}")

    # A composite that rewards durability and growth and only mildly rewards yield, so a
    # collapsing-price high-yield name cannot rank first.
    z = lambda s: (s - s.mean()) / s.std(ddof=0)      # noqa: E731
    f = f.assign(score=(1.2 * z(f["roe_mean5"].clip(-0.1, 0.4))
                        + 1.0 * z(f["dps_cagr5"].clip(-0.3, 0.6).fillna(0))
                        + 0.8 * z(f["streak"].astype(float))
                        + 0.6 * z(f["yield"].clip(0, 0.12))
                        - 0.8 * z(f["roe_std5"].fillna(f["roe_std5"].median()))
                        - 0.6 * z(f["dps_cuts_6y"].astype(float))))
    f = f.sort_values("score", ascending=False)
    atomic_to_parquet(f.reset_index(), OUT / "dividend_stage1.parquet", index=False)

    print(f"\n{'=' * 104}\nTOP 45 BY DURABILITY COMPOSITE (not by yield -- yield is a floor only)"
          f"\n{'=' * 104}")
    print(f"  {'code':>12} {'name':<10} {'sec':>5} {'yield':>7} {'DPS cagr5':>10} {'streak':>7} "
          f"{'cuts':>5} {'ROE5':>7} {'ROEmin':>7} {'ROEsd':>7} {'score':>6}")
    for code, r in f.head(45).iterrows():
        print(f"  {code:>12} {str(r['name'])[:10]:<10} {str(r['csrc']):>5} {r['yield']:>7.2%} "
              f"{r['dps_cagr5']:>10.1%} {int(r['streak']):>7} {int(r['dps_cuts_6y']):>5} "
              f"{r['roe_mean5']:>7.1%} {r['roe_min5']:>7.1%} {r['roe_std5']:>7.1%} "
              f"{r['score']:>6.2f}")
    print("\n  sector spread of the top 45: ")
    print("   ", f.head(45)["bucket"].value_counts().to_dict())
    print(f"\nwrote {OUT / 'dividend_stage1.parquet'}  ({len(f)} survivors)")


if __name__ == "__main__":
    main()
