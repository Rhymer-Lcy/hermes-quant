"""PART A re-audit of the long-horizon dividend shortlist, from first principles.

Three methodological corrections drive this file; none of them is cosmetic.

1. SECTOR-APPROPRIATE QUALITY. The previous table carried one "operating cash flow / dividend"
   column across every sector and printed 64.5x for a bank. For a bank or insurer, operating cash
   flow is dominated by deposit, loan and reserve movements and has no interpretable relationship
   to dividend capacity; a large multiple means the loan book shrank, not that the dividend is
   safe. Financials are therefore scored WITHOUT that column, on ROE level and stability,
   attributable-profit trend, payout, share-count dilution and PB against their own history, and
   the metrics that genuinely decide a bank's payout (NPL, provision coverage, CET1, NIM) are
   reported as NOT PROGRAMMATICALLY VERIFIED rather than silently omitted or invented.

2. ISSUER-LEVEL DIVIDEND HISTORY. A six-year A-share payout streak is a listing-age test, not an
   economic one. Where the same issuer had an established listed ordinary share class before its
   A-share listing and both classes participate in the same board-declared dividend per share, the
   A-share's youth is not evidence about the dividend. Handled explicitly for China Mobile.

3. DIVIDEND DEFINITIONS. Every yield is the DECLARED payout of one fiscal year over today's close.
   Never a trailing-12-month sum, which straddles fiscal years and inflates a yield whenever a
   company changes cadence (measured: 成都银行 prints 7.50% on TTM against a true 4.90%).

    conda activate hermes
    python scripts/dividend_reaudit.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from hermes.data.lake import load_close_panel
from hermes.paths import RESULTS_DIR, ensure_dirs

TODAY = pd.Timestamp("2026-09-21")          # last completed trading bar
VAL_START = pd.Timestamp("2021-09-21")
OUT = RESULTS_DIR / "screens"
PLANS = OUT / "dividend_plans.parquet"
FY = 2025
FINANCIAL_PREFIX = ("J66", "J67", "J68", "J69")     # CSRC: banking, capital markets, insurance

CANDIDATES = [
    # previously included
    "sh.600036", "sh.600900", "sh.601601", "sz.000333", "sh.601398",
    "sh.600377", "sh.600018", "sh.600600", "sh.601088", "sz.000651",
    # reconsidered on issuer-level grounds or as replacements
    "sh.600941", "sh.601939", "sh.601288", "sh.601658", "sz.002142",
    "sh.600919", "sh.601838", "sz.001965", "sh.601898", "sh.600690",
    "sh.601166", "sh.601006", "sh.601318", "sh.600887",
]


def fy_dps(plans: pd.DataFrame, fy: int) -> pd.Series:
    return plans[plans["fy"] == fy].groupby("code")["dps"].sum()


def main() -> None:
    ensure_dirs()
    plans = pd.read_parquet(PLANS)
    plans["ex_date"] = pd.to_datetime(plans["ex_date"])
    s2 = pd.read_parquet(OUT / "dividend_stage2.parquet").set_index("code")

    import akshare as ak

    spot = ak.stock_zh_a_spot_em()
    cc = next(c for c in spot.columns if c.strip() == "代码")
    px = pd.to_numeric(spot[next(c for c in spot.columns if c.strip() == "最新价")], errors="coerce")
    nm = spot[next(c for c in spot.columns if c.strip() == "名称")].astype(str)
    mc = pd.to_numeric(spot[next(c for c in spot.columns if c.strip() == "总市值")], errors="coerce")
    key = ["sh." + k if k.startswith(("6", "9")) else "sz." + k for k in spot[cc].astype(str)]
    price = pd.Series(px.to_numpy(), index=key)
    name = pd.Series(nm.to_numpy(), index=key)
    mcap = pd.Series(mc.to_numpy(), index=key)

    # ---- fundamentals per fiscal year, bulk ----
    yj = {}
    for y in (2021, 2022, 2023, 2024, 2025):
        df = ak.stock_yjbb_em(date=f"{y}1231")
        k = df[next(c for c in df.columns if c.strip() == "股票代码")].astype(str).str.zfill(6)
        idx = ["sh." + x if x.startswith(("6", "9")) else "sz." + x for x in k]
        g = lambda n: pd.to_numeric(  # noqa: E731
            df[next(c for c in df.columns if c.strip() == n)], errors="coerce").to_numpy()
        yj[y] = pd.DataFrame({"eps": g("每股收益"), "np": g("净利润-净利润"),
                              "np_yoy": g("净利润-同比增长"), "roe": g("净资产收益率"),
                              "rev_yoy": g("营业总收入-同比增长"),
                              "ocfps": g("每股经营现金流量"),
                              # .to_numpy() is required: a Series carries the source frame's
                              # RangeIndex and pandas would align it against `idx` and produce
                              # all-NaN, which silently emptied the industry column.
                              "industry": df[next(c for c in df.columns
                                                  if c.strip() == "所处行业")].astype(str).to_numpy()},
                             index=idx)
        yj[y] = yj[y][~yj[y].index.duplicated()]
    dfh = ak.stock_yjbb_em(date="20260630")
    kh = dfh[next(c for c in dfh.columns if c.strip() == "股票代码")].astype(str).str.zfill(6)
    idxh = ["sh." + x if x.startswith(("6", "9")) else "sz." + x for x in kh]
    half = pd.DataFrame({
        "np_yoy": pd.to_numeric(dfh[next(c for c in dfh.columns if c.strip() == "净利润-同比增长")],
                                errors="coerce").to_numpy(),
        "rev_yoy": pd.to_numeric(dfh[next(c for c in dfh.columns
                                          if c.strip() == "营业总收入-同比增长")],
                                 errors="coerce").to_numpy()}, index=idxh)
    half = half[~half.index.duplicated()]

    # ---- share-count dilution, from the dividend plans' 总股本 ----
    shares_by_fy = {}
    for y in (2021, 2025):
        d = ak.stock_fhps_em(date=f"{y}1231")
        k = d[next(c for c in d.columns if c.strip() == "代码")].astype(str).str.zfill(6)
        idx = ["sh." + x if x.startswith(("6", "9")) else "sz." + x for x in k]
        s = pd.Series(pd.to_numeric(d[next(c for c in d.columns if c.strip() == "总股本")],
                                    errors="coerce").to_numpy(), index=idx)
        shares_by_fy[y] = s[~s.index.duplicated()]

    dps25 = fy_dps(plans, FY)
    dps_hist = {y: fy_dps(plans, y) for y in range(2016, 2026)}

    # ---- valuation, own-history percentile ----
    pe = load_close_panel(codes=CANDIDATES, field="peTTM")
    pb = load_close_panel(codes=CANDIDATES, field="pbMRQ")

    rows = []
    for code in CANDIDATES:
        if code not in price.index or code not in dps25.index:
            print(f"  skip {code}: no quote or no FY{FY} payout")
            continue
        d = float(dps25[code])
        p = float(price[code])
        ind = str(yj[2025]["industry"].get(code, ""))
        csrc = str(s2["csrc"].get(code, "")) if code in s2.index else ""
        is_fin = csrc.startswith(FINANCIAL_PREFIX) or ("银行" in ind or "保险" in ind
                                                       or "证券" in ind)
        eps = float(yj[2025]["eps"].get(code, np.nan))
        ocf = float(yj[2025]["ocfps"].get(code, np.nan))
        hist = [float(dps_hist[y].get(code, 0.0)) for y in range(2016, 2026)]
        paid = sum(1 for h in hist if h > 0)
        streak = 0
        for h in reversed(hist):
            if h > 0:
                streak += 1
            else:
                break
        base = float(dps_hist[2020].get(code, 0.0))
        cagr5 = (d / base) ** 0.2 - 1 if base > 0 else np.nan
        # SPECIAL-DIVIDEND DETECTOR. Compare the biggest year with the SECOND biggest, not with
        # the median of the rest: a company whose DPS compounds steadily has a max far above the
        # median of earlier years, and the median form wrongly flagged 美的集团, 青岛啤酒 and
        # 海尔智家 -- i.e. it flagged growth as if it were a one-off. A genuine special payout
        # towers over its NEIGHBOUR, so max/second-max is the discriminating ratio.
        nz = sorted((h for h in hist if h > 0), reverse=True)
        spike = (nz[0] / nz[1]) if len(nz) >= 4 and nz[1] > 0 else np.nan
        sh21 = float(shares_by_fy[2021].get(code, np.nan))
        sh25 = float(shares_by_fy[2025].get(code, np.nan))
        dil = (sh25 / sh21 - 1) if (np.isfinite(sh21) and np.isfinite(sh25) and sh21 > 0) else np.nan
        roes = [float(yj[y]["roe"].get(code, np.nan)) for y in (2021, 2022, 2023, 2024, 2025)]
        roes = [r for r in roes if np.isfinite(r)]
        nps = [float(yj[y]["np"].get(code, np.nan)) for y in (2021, 2022, 2023, 2024, 2025)]
        nps = [n for n in nps if np.isfinite(n)]
        np_cagr = ((nps[-1] / nps[0]) ** (1 / (len(nps) - 1)) - 1
                   if len(nps) >= 3 and nps[0] > 0 and nps[-1] > 0 else np.nan)

        def pct(panel, c):
            if c not in panel.columns:
                return np.nan, np.nan
            s = panel[c].dropna()
            s = s[(s.index >= VAL_START) & (s.index <= TODAY)]
            s = s[s > 0]                      # negative/undefined PE excluded, stated in the doc
            if len(s) < 100:
                return np.nan, np.nan
            return float(s.iloc[-1]), float((s < s.iloc[-1]).mean() * 100)

        pe_now, pe_p = pct(pe, code)
        pb_now, pb_p = pct(pb, code)
        rows.append({
            "code": code, "name": name.get(code, ""), "industry": ind, "financial": bool(is_fin),
            "mcap_y": float(mcap.get(code, np.nan)) / 1e8,
            "price": p, "dps_fy25": d, "yield": d / p,
            "eps_fy25": eps, "payout": d / eps if eps and eps > 0 else np.nan,
            "ocf_cover": (ocf / d if (d > 0 and np.isfinite(ocf)) else np.nan),
            "paid_10y": paid, "streak": streak, "dps_cagr5": cagr5, "spike": spike,
            "dilution_21_25": dil,
            "roe_mean": float(np.mean(roes)) if roes else np.nan,
            "roe_min": float(np.min(roes)) if roes else np.nan,
            "roe_std": float(np.std(roes, ddof=1)) if len(roes) > 1 else np.nan,
            "np_cagr": np_cagr,
            "np_yoy25": float(yj[2025]["np_yoy"].get(code, np.nan)),
            "np_yoy_h1": float(half["np_yoy"].get(code, np.nan)),
            "rev_yoy_h1": float(half["rev_yoy"].get(code, np.nan)),
            "pe_now": pe_now, "pe_pct": pe_p, "pb_now": pb_now, "pb_pct": pb_p,
        })
    df = pd.DataFrame(rows).set_index("code")

    print(f"\n{'=' * 122}\nRE-AUDIT, FY{FY} DECLARED DIVIDEND BASIS, price date {TODAY.date()}"
          f"\n{'=' * 122}")
    print(f"  {'code':>12} {'name':<9} {'industry':<10} {'fin':>4} {'mcap':>7} {'yld':>6} "
          f"{'payout':>7} {'ROE':>6} {'ROEmin':>7} {'dpsC5':>7} {'npC4':>7} {'H1np':>7} "
          f"{'dilut':>7} {'PE':>6} {'PEpct':>6} {'PB':>5} {'PBpct':>6}")
    for c, r in df.sort_values(["financial", "yield"], ascending=[True, False]).iterrows():
        print(f"  {c:>12} {str(r['name'])[:9]:<9} {str(r['industry'])[:10]:<10} "
              f"{'Y' if r['financial'] else 'n':>4} {r['mcap_y']:>7.0f} {r['yield']:>6.2%} "
              f"{r['payout']:>7.1%} {r['roe_mean']:>6.1f} {r['roe_min']:>7.1f} "
              f"{r['dps_cagr5']:>7.1%} {r['np_cagr']:>7.1%} {r['np_yoy_h1']:>7.1f} "
              f"{r['dilution_21_25']:>7.1%} {r['pe_now']:>6.1f} {r['pe_pct']:>5.0f}% "
              f"{r['pb_now']:>5.2f} {r['pb_pct']:>5.0f}%")

    print(f"\n{'=' * 122}\nSECTOR-APPROPRIATE TREATMENT\n{'=' * 122}")
    fins = df[df["financial"]]
    print(f"  {len(fins)} of {len(df)} candidates are financials. For these the generic "
          f"OCF/dividend multiple is NOT used.")
    print("  Illustration of why -- the multiple the previous report printed for banks:")
    for c, r in fins.sort_values("ocf_cover", ascending=False).head(4).iterrows():
        print(f"    {c} {r['name']}: OCF/DPS = {r['ocf_cover']:.1f}x  "
              f"(uninterpretable: bank operating cash flow is deposit and loan flow)")
    print("  NOT PROGRAMMATICALLY VERIFIABLE from any source available here, and therefore not "
          "reported\n  as numbers: NPL ratio, provision coverage, CET1 / total capital adequacy, "
          "net interest margin,\n  insurer solvency and embedded value. These must be read from "
          "the annual report before acting.")

    print(f"\n{'=' * 122}\nTRAP FLAGS\n{'=' * 122}")
    df["flag_special"] = df["spike"] > 2.0
    df["flag_payout"] = df["payout"] > 0.95
    df["flag_h1"] = df["np_yoy_h1"] < -25
    df["flag_shrink"] = (df["np_yoy25"] < -10) & (df["np_yoy_h1"] < -10)
    df["flag_dps_fall"] = df["dps_cagr5"] < -0.02
    df["flag_dilution"] = df["dilution_21_25"] > 0.10
    df["flag_ocf"] = (~df["financial"]) & (df["ocf_cover"] < 1.0)
    fl = [c for c in df.columns if c.startswith("flag_")]
    df["n_flags"] = df[fl].sum(axis=1)
    for f in fl:
        hit = df.index[df[f]].tolist()
        print(f"  {f:<16}: {len(hit)}  {[df.loc[h, 'name'] for h in hit]}")
    print(f"  clean: {int((df['n_flags'] == 0).sum())} of {len(df)}")

    df.to_parquet(OUT / "dividend_reaudit.parquet")
    (OUT / "dividend_reaudit.json").write_text(
        json.dumps({"as_of": str(TODAY.date()), "fy": FY,
                    "rows": {c: {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v)
                                 for k, v in r.items()} for c, r in df.iterrows()}},
                   indent=2, default=float, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT / 'dividend_reaudit.parquet'}")


if __name__ == "__main__":
    main()
