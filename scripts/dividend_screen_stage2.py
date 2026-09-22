"""Stage 2 of the long-horizon A-share dividend screen: verify stage-1 survivors against
reported statements, and kill the dividend traps stage 1 cannot see.

Stage 1 ranked on payout durability from the local lake. That is not enough, and its composite
has a known bias: a ONE-OFF SPECIAL dividend simultaneously inflates the printed yield and the
DPS growth rate, so the very names a five-year holder should avoid float to the top. This stage
therefore does three things stage 1 cannot:

  1. detects one-off / special payouts (a cycle whose DPS towers over its own neighbours)
  2. reads actual coverage -- EPS and operating cash flow per share against DPS, and the
     company's own reported payout ratio
  3. reads leverage, margin and growth trend, so a shrinking or debt-funded payer is visible

Data comes from Eastmoney BULK report-date endpoints (one call per report period, not per stock),
which are the same filings the exchanges publish. Everything is re-checked by hand for the final
shortlist -- a screen is a funnel, not a verdict.

    conda activate hermes
    python scripts/dividend_screen_stage2.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.io import atomic_to_parquet
from hermes.paths import RESULTS_DIR, ensure_dirs

OUT = RESULTS_DIR / "screens"
FY = [2021, 2022, 2023, 2024, 2025]
HALF = "20260630"


def _col(df: pd.DataFrame, *wants: str) -> str:
    for w in wants:
        for c in df.columns:
            if c.strip() == w:
                return c
    raise KeyError(f"none of {wants} in {[c.strip() for c in df.columns]}")


def _key(code: str) -> str:
    return code.split(".")[1]


def pull_yjbb(date: str) -> pd.DataFrame:
    import akshare as ak

    df = ak.stock_yjbb_em(date=date)
    out = pd.DataFrame({
        "k": df[_col(df, "股票代码")].astype(str),
        "eps": pd.to_numeric(df[_col(df, "每股收益")], errors="coerce"),
        "rev": pd.to_numeric(df[_col(df, "营业总收入-营业总收入")], errors="coerce"),
        "rev_yoy": pd.to_numeric(df[_col(df, "营业总收入-同比增长")], errors="coerce"),
        "np": pd.to_numeric(df[_col(df, "净利润-净利润")], errors="coerce"),
        "np_yoy": pd.to_numeric(df[_col(df, "净利润-同比增长")], errors="coerce"),
        "bps": pd.to_numeric(df[_col(df, "每股净资产")], errors="coerce"),
        "roe": pd.to_numeric(df[_col(df, "净资产收益率")], errors="coerce"),
        "ocfps": pd.to_numeric(df[_col(df, "每股经营现金流量")], errors="coerce"),
        "gross": pd.to_numeric(df[_col(df, "销售毛利率")], errors="coerce"),
        "industry": df[_col(df, "所处行业")].astype(str),
    })
    return out.drop_duplicates("k").set_index("k")


def pull_fhps(date: str) -> pd.DataFrame:
    import akshare as ak

    df = ak.stock_fhps_em(date=date)
    out = pd.DataFrame({
        "k": df[_col(df, "代码")].astype(str),
        "cash_per10": pd.to_numeric(df[_col(df, "现金分红-现金分红比例")], errors="coerce"),
        "payout": pd.to_numeric(df[_col(df, "现金分红-股息率")], errors="coerce"),
        "shares": pd.to_numeric(df[_col(df, "总股本")], errors="coerce"),
    })
    return out.drop_duplicates("k").set_index("k")


def pull_zcfz(date: str) -> pd.DataFrame:
    import akshare as ak

    df = ak.stock_zcfz_em(date=date)
    try:
        lev = _col(df, "资产负债率")
    except KeyError:
        return pd.DataFrame(columns=["debt_ratio"]).rename_axis("k")
    out = pd.DataFrame({"k": df[_col(df, "股票代码")].astype(str),
                        "debt_ratio": pd.to_numeric(df[lev], errors="coerce")})
    return out.drop_duplicates("k").set_index("k")


def main() -> None:
    ensure_dirs()
    s1 = pd.read_parquet(OUT / "dividend_stage1.parquet").set_index("code")
    print(f"stage-1 survivors: {len(s1)}")

    yj = {y: pull_yjbb(f"{y}1231") for y in FY}
    for y in FY:
        print(f"  yjbb FY{y}: {len(yj[y])} rows")
    yj_h = pull_yjbb(HALF)
    print(f"  yjbb {HALF}: {len(yj_h)} rows")
    fh = {y: pull_fhps(f"{y}1231") for y in FY}
    for y in FY:
        print(f"  fhps FY{y}: {len(fh[y])} rows")
    try:
        zc = pull_zcfz(HALF)
        print(f"  zcfz {HALF}: {len(zc)} rows")
    except Exception as e:                      # noqa: BLE001 -- leverage is a nice-to-have
        print(f"  zcfz: unavailable ({type(e).__name__}); leverage left blank")
        zc = pd.DataFrame(columns=["debt_ratio"]).rename_axis("k")

    rows = []
    for code, r in s1.iterrows():
        k = _key(code)
        dps_hist = {y: float(fh[y]["cash_per10"].get(k, np.nan)) / 10.0 for y in FY}
        eps_hist = {y: float(yj[y]["eps"].get(k, np.nan)) for y in FY}
        ocf_hist = {y: float(yj[y]["ocfps"].get(k, np.nan)) for y in FY}
        roe_hist = {y: float(yj[y]["roe"].get(k, np.nan)) for y in FY}
        np_hist = {y: float(yj[y]["np"].get(k, np.nan)) for y in FY}

        d = pd.Series(dps_hist).dropna()
        # SPECIAL-DIVIDEND DETECTOR: a cycle whose DPS is far above the median of the others.
        spike = np.nan
        if len(d) >= 4:
            top_y = d.idxmax()
            others = d.drop(index=top_y)
            spike = float(d.max() / others.median()) if others.median() > 0 else np.inf
        # DPS SOURCE MATTERS. Eastmoney's FY-annual plan carries only the FINAL instalment, so for
        # a company that also pays an interim it understates the year's cash by roughly half --
        # verified on sh.600036 (ex 2026-01-16 1.0130 interim + ex 2026-07-10 1.0030 final =
        # 2.0160, against an FY2025 plan figure of 1.0030) and on sh.600919 / sh.601166, while
        # single-payment names such as sh.601838 and sh.600600 agree exactly. Understating DPS
        # here would understate payout and overstate coverage for precisely the large banks that
        # dominate this shortlist, so the CALENDAR-CYCLE sum from the local ex-date lake is used:
        # for these payers the calendar-2026 ex-dates are the FY2025 interim plus the FY2025 final,
        # which is the right counterpart to FY2025 EPS.
        dps25 = float(r["dps_latest"])
        eps25, ocf25 = eps_hist.get(2025), ocf_hist.get(2025)
        dps_final_only = dps_hist.get(2025)
        payout = dps25 / eps25 if (eps25 and eps25 > 0) else np.nan
        ocf_cov = ocf25 / dps25 if (dps25 and dps25 > 0) else np.nan
        interim_payer = (np.isfinite(dps_final_only or np.nan) and dps_final_only > 0
                         and dps25 / dps_final_only > 1.3)
        npv = pd.Series(np_hist).dropna()
        np_cagr = ((npv.iloc[-1] / npv.iloc[0]) ** (1 / (len(npv) - 1)) - 1
                   if len(npv) >= 3 and npv.iloc[0] > 0 and npv.iloc[-1] > 0 else np.nan)
        h = yj_h.loc[k] if k in yj_h.index else None
        rows.append({
            "code": code, "name": r["name"], "industry": yj[2025]["industry"].get(k, ""),
            "csrc": r["csrc"], "bucket": r["bucket"], "price": r["price"],
            "yield": r["yield"], "dps_latest": r["dps_latest"],
            "streak": r["streak"], "dps_cagr5": r["dps_cagr5"], "dps_cuts_6y": r["dps_cuts_6y"],
            "roe_mean5": r["roe_mean5"], "roe_min5": r["roe_min5"], "roe_std5": r["roe_std5"],
            "spike_ratio": spike,
            "payout_fy25": payout, "ocf_cover_fy25": ocf_cov,
            "eps_fy25": eps25, "ocfps_fy25": ocf25, "dps_fy25": dps25,
            "dps_final_only": dps_final_only, "interim_payer": bool(interim_payer),
            "roe_fy25": roe_hist.get(2025), "gross_fy25": float(yj[2025]["gross"].get(k, np.nan)),
            "np_cagr_4y": np_cagr,
            "np_yoy_fy25": float(yj[2025]["np_yoy"].get(k, np.nan)),
            "rev_yoy_fy25": float(yj[2025]["rev_yoy"].get(k, np.nan)),
            "np_yoy_h1": float(h["np_yoy"]) if h is not None else np.nan,
            "rev_yoy_h1": float(h["rev_yoy"]) if h is not None else np.nan,
            "roe_h1": float(h["roe"]) if h is not None else np.nan,
            "debt_ratio": float(zc["debt_ratio"].get(k, np.nan)) if len(zc) else np.nan,
        })
    df = pd.DataFrame(rows).set_index("code")

    # ---- trap flags: each is a REASON TO EXCLUDE, reported rather than silently applied ----
    df["flag_special"] = df["spike_ratio"] > 2.0
    df["flag_payout"] = df["payout_fy25"] > 0.95
    df["flag_ocf"] = df["ocf_cover_fy25"] < 1.0
    df["flag_shrink"] = (df["np_yoy_fy25"] < -10) & (df["np_yoy_h1"] < -10)
    df["flag_h1_collapse"] = df["np_yoy_h1"] < -25
    df["flag_lev"] = (df["debt_ratio"] > 70) & (~df["csrc"].astype(str).str.startswith("J"))
    flags = ["flag_special", "flag_payout", "flag_ocf", "flag_shrink", "flag_h1_collapse",
             "flag_lev"]
    df["n_flags"] = df[flags].sum(axis=1)

    print(f"\n{'=' * 118}\nTRAP FLAGS across {len(df)} stage-1 survivors\n{'=' * 118}")
    for f in flags:
        print(f"  {f:<18}: {int(df[f].sum()):>3} names")
    print(f"  clean (0 flags)  : {int((df['n_flags'] == 0).sum()):>3} names")

    clean = df[df["n_flags"] == 0].copy()
    atomic_to_parquet(df.reset_index(), OUT / "dividend_stage2.parquet", index=False)

    print(f"\n{'=' * 118}\nTOP STAGE-1 NAMES AND WHY THEY SURVIVE OR DIE\n{'=' * 118}")
    print(f"  {'code':>12} {'name':<9} {'yld':>6} {'spike':>6} {'payout':>7} {'ocfcov':>7} "
          f"{'ROE25':>6} {'npY25':>7} {'npYH1':>7} {'debt':>6} {'flags':>5}")
    for code, r in df.sort_values("yield", ascending=False).head(28).iterrows():
        fl = ",".join(f.replace("flag_", "") for f in flags if r[f]) or "-"
        print(f"  {code:>12} {str(r['name'])[:9]:<9} {r['yield']:>6.2%} "
              f"{r['spike_ratio']:>6.2f} {r['payout_fy25']:>7.2f} {r['ocf_cover_fy25']:>7.2f} "
              f"{r['roe_fy25']:>6.1f} {r['np_yoy_fy25']:>7.1f} {r['np_yoy_h1']:>7.1f} "
              f"{r['debt_ratio']:>6.1f}  {fl}")

    print(f"\n{'=' * 118}\nCLEAN CANDIDATES ({len(clean)}) sorted by yield\n{'=' * 118}")
    print(f"  {'code':>12} {'name':<9} {'industry':<12} {'yld':>6} {'payout':>7} {'ocfcov':>7} "
          f"{'ROE5':>6} {'ROEmin':>7} {'dpsCAGR':>8} {'npCAGR':>7} {'npYH1':>7} {'debt':>6}")
    for code, r in clean.sort_values("yield", ascending=False).iterrows():
        print(f"  {code:>12} {str(r['name'])[:9]:<9} {str(r['industry'])[:12]:<12} "
              f"{r['yield']:>6.2%} {r['payout_fy25']:>7.2f} {r['ocf_cover_fy25']:>7.2f} "
              f"{r['roe_mean5']:>6.1%} {r['roe_min5']:>7.1%} {r['dps_cagr5']:>8.1%} "
              f"{r['np_cagr_4y']:>7.1%} {r['np_yoy_h1']:>7.1f} {r['debt_ratio']:>6.1f}")
    print(f"\nwrote {OUT / 'dividend_stage2.parquet'}")


if __name__ == "__main__":
    main()
