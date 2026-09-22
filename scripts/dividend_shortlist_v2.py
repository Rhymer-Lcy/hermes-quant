"""The re-audited long-horizon dividend shortlist (PART A7/A8). Supersedes the first pass.

Selection is a judgement made ON TOP of the verified re-audit table, not a formula. What changed
since the first list, and why, is recorded explicitly so the revision can be argued with.

    conda activate hermes
    python scripts/dividend_shortlist_v2.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from hermes.paths import RESULTS_DIR, ensure_dirs

OUT = RESULTS_DIR / "screens"
TODAY = pd.Timestamp("2026-09-21")

# (code, bucket, thesis, invalidation)
FINAL = [
    ("sh.600036", "A", "retail deposit franchise, best-in-class bank ROE, no dilution in five years",
     "ROE below 12%, payout pushed past 40% to hold the DPS, or any equity raise"),
    ("sh.601601", "A", "insurer with the lowest payout here (20.7%) and PE in its own 2nd percentile",
     "solvency pressure forcing a payout cut, or reliance on investment gains for profit growth"),
    ("sz.002142", "A", "fastest-growing bank in the set with ZERO share-count dilution",
     "the growth premium is funded by a capital raise; watch share count every year"),
    ("sh.600900", "A", "cascade hydro, near-annuity output, regulated returns",
     "a new capex cycle, or payout pushed past 80% to cover a weak hydrology year"),
    ("sz.000333", "A", "global appliance platform, 20%+ ROE through a property downturn",
     "payout above 80% while profit growth stalls; PB already at its 80th percentile"),
    ("sh.600600", "B", "beer premiumisation, staple demand, cheapest name here vs its own history",
     "volume decline turning the premiumisation story into a price-only story"),
    ("sh.600690", "B", "high-ROE appliance maker at PE 4th / PB 2nd percentile, buying back stock",
     "the 2026H1 profit fall (-14.3%) extending into a second and third half-year"),
    ("sh.600377", "B", "Jiangsu toll concessions, low capex intensity, no dilution",
     "concession expiry without replacement, or a debt-funded acquisition"),
    ("sh.600018", "B", "Shanghai port throughput, 33% payout, PB in its 12th percentile",
     "sustained throughput decline, or payout ratio rising to prop a falling DPS"),
    ("sh.601088", "C", "coal plus captive rail and power; real cash, but the payout tracks coal",
     "coal price mean-reverting while the 75.6% payout is held; PE/PB already 93rd/96th pct"),
    ("sh.601398", "C", "mega-bank, policy-anchored payout -- held for yield, NOT for value",
     "PE 96th / PB 93rd percentile is the thesis risk; a de-rating costs years of dividend"),
]
WATCHLIST = {
    "sh.600941": "中国移动 -- the previous exclusion reason was WRONG (see below). Profit is "
                 "positive and the issuer's dividend record is long. Held back only because "
                 "FY2025 profit growth is -0.90% and 2026H1 -6.30% while the payout is already "
                 "74.1%, so future DPS growth must come from a payout ratio with little room.",
    "sh.601318": "中国平安 -- PE in its own 0th percentile and a 4.98% yield is genuinely "
                 "attractive, but 2026H1 profit +36.1% is investment-income driven and the set "
                 "already carries one insurer with a lower payout.",
}
REMOVED = {
    "sz.000651": "格力电器 -- REMOVED. On the corrected fiscal-year dividend basis its five-year "
                 "DPS CAGR is -5.6%, i.e. the dividend per share is FALLING, not flat as the "
                 "previous report said. 2026H1 profit -7.9%. A falling DPS disqualifies it from a "
                 "5-10 year dividend thesis whatever the 7.87% headline yield suggests.",
}
EXCLUDED = {
    "sh.601006": "大秦铁路 -- DPS CAGR -14.3%, four-year profit CAGR -16.6%, AND share count up "
                 "33.6%. Every one of the three moves the wrong way.",
    "sh.601658": "邮储银行 -- share count +30.0% over FY2021-25. Per-share dividend growth of "
                 "0.9%/yr against that dilution is not a compounding story.",
    "sh.600919": "江苏银行 -- share count +24.3%, and PE/PB at the 94th/93rd percentile of its "
                 "own history.",
    "sh.601838": "成都银行 -- strong ROE and growth, but share count +17.3%; kept out in favour "
                 "of 宁波银行, which delivers comparable growth with zero dilution.",
    "sh.601939": "建设银行 -- PE 99th and PB 99th percentile of its own five-year history. The "
                 "most extended valuation in the whole candidate set.",
    "sh.601288": "农业银行 -- PE 87th / PB 85th percentile; same crowding risk, less franchise "
                 "quality than 招商银行.",
    "sh.601166": "兴业银行 -- 5.99% yield but four-year profit CAGR -1.6% and 2026H1 -4.7%.",
    "sh.600887": "伊利股份 -- 2026H1 profit -20.0% with a 75.4% payout.",
    "sz.001965": "招商公路 -- four-year profit CAGR -1.9% and PE at its 85th percentile.",
    "sh.601898": "中煤能源 -- cyclical, and PE/PB at the 83rd/77th percentile.",
}


def main() -> None:
    ensure_dirs()
    df = pd.read_parquet(OUT / "dividend_reaudit.parquet")
    codes = [c for c, _b, _t, _i in FINAL]
    missing = [c for c in codes if c not in df.index]
    if missing:
        raise SystemExit(f"ABORT: shortlist names absent from the re-audit: {missing}")
    sel = df.loc[codes].copy()
    sel["bucket"] = [b for _c, b, _t, _i in FINAL]
    sel["thesis"] = [t for _c, _b, t, _i in FINAL]
    sel["invalidation"] = [i for _c, _b, _t, i in FINAL]

    # ---- gates: nothing with a falling DPS, heavy dilution or a broken streak may appear ----
    bad_dps = sel.index[sel["dps_cagr5"] < -0.02].tolist()
    bad_dil = sel.index[sel["dilution_21_25"] > 0.10].tolist()
    if bad_dps:
        raise SystemExit(f"ABORT: falling-DPS names on the shortlist: {bad_dps}")
    if bad_dil:
        raise SystemExit(f"ABORT: heavily diluting names on the shortlist: {bad_dil}")
    if int(sel["streak"].min()) < 6:
        raise SystemExit("ABORT: a shortlist name has a payout streak under six years")

    labels = {"A": "A. DURABLE DIVIDEND COMPOUNDERS", "B": "B. STABLE YIELD / LOWER GROWTH",
              "C": "C. CYCLICAL OR VALUATION-SENSITIVE"}
    for b, label in labels.items():
        g = sel[sel["bucket"] == b]
        print(f"\n{'=' * 124}\n{label}\n{'=' * 124}")
        print(f"  {'code':>12} {'name':<9} {'px':>8} {'yield':>6} {'payout':>7} {'ROE':>6} "
              f"{'ROEmin':>7} {'dpsC5':>7} {'npC4':>7} {'H1np':>7} {'dilut':>7} "
              f"{'PE':>6} {'PEpct':>6} {'PB':>5} {'PBpct':>6} {'fin':>4}")
        for c, r in g.iterrows():
            print(f"  {c:>12} {str(r['name'])[:9]:<9} {r['price']:>8.2f} {r['yield']:>6.2%} "
                  f"{r['payout']:>7.1%} {r['roe_mean']:>6.1f} {r['roe_min']:>7.1f} "
                  f"{r['dps_cagr5']:>7.1%} {r['np_cagr']:>7.1%} {r['np_yoy_h1']:>7.1f} "
                  f"{r['dilution_21_25']:>7.1%} {r['pe_now']:>6.1f} {r['pe_pct']:>5.0f}% "
                  f"{r['pb_now']:>5.2f} {r['pb_pct']:>5.0f}% {'Y' if r['financial'] else 'n':>4}")
        for c, r in g.iterrows():
            print(f"      {r['name']}: {r['thesis']}")
            print(f"        invalidated if: {r['invalidation']}")

    # ---- A8 portfolio-level sanity ----
    print(f"\n{'=' * 124}\nA8  PORTFOLIO-LEVEL SANITY (equal weight, research shortlist -- "
          f"NOT an allocation)\n{'=' * 124}")
    fin = int(sel["financial"].sum())
    banks = int(sel["industry"].astype(str).str.contains("银行").sum())
    ind = sel["industry"].astype(str).value_counts()
    print(f"  names                       : {len(sel)}")
    print(f"  distinct industries         : {sel['industry'].nunique()}  {dict(ind)}")
    print(f"  financials                  : {fin}/{len(sel)} ({fin / len(sel):.0%}), "
          f"of which banks {banks}")
    print(f"  largest single industry     : {ind.index[0]} at {ind.iloc[0]}/{len(sel)} "
          f"({ind.iloc[0] / len(sel):.0%} equal-weighted)")
    print(f"  equal-weight dividend yield : {sel['yield'].mean():.2%}")
    print(f"  payout ratio                : min {sel['payout'].min():.1%}  median "
          f"{sel['payout'].median():.1%}  max {sel['payout'].max():.1%}")
    hi = sel[sel["pe_pct"] > 80]
    lo = sel[sel["pe_pct"] < 20]
    print(f"  valuation concentration     : {len(hi)}/{len(sel)} above their own 80th PE "
          f"percentile ({list(hi['name'])}); {len(lo)}/{len(sel)} below their 20th "
          f"({list(lo['name'])})")
    print(f"  cyclical exposure           : {int((sel['bucket'] == 'C').sum())}/{len(sel)} "
          f"in the cyclical / valuation-sensitive bucket")
    soe = ["sh.600036", "sh.601601", "sh.600900", "sh.600600", "sh.600377", "sh.600018",
           "sh.601088", "sh.601398"]
    n_soe = sum(1 for c in sel.index if c in soe)
    print(f"  state-owned / state-linked  : {n_soe}/{len(sel)} ({n_soe / len(sel):.0%}); "
          f"privately controlled: {list(sel.loc[[c for c in sel.index if c not in soe], 'name'])}")
    print(f"  names with 2026H1 profit down: {int((sel['np_yoy_h1'] < 0).sum())}/{len(sel)} "
          f"({list(sel.loc[sel['np_yoy_h1'] < 0, 'name'])})")
    print(f"  zero-dilution names          : {int((sel['dilution_21_25'] <= 0.001).sum())}"
          f"/{len(sel)}")

    print(f"\n{'=' * 124}\nPREVIOUSLY INCLUDED, NOW REMOVED\n{'=' * 124}")
    for c, why in REMOVED.items():
        print(f"  {c}: {why}")
    print(f"\n{'=' * 124}\nPREVIOUSLY EXCLUDED, NOW RECONSIDERED\n{'=' * 124}")
    for c, why in WATCHLIST.items():
        print(f"  {c}: {why}")
    print(f"\n{'=' * 124}\nEXAMINED AND EXCLUDED\n{'=' * 124}")
    for c, why in EXCLUDED.items():
        print(f"  {c}: {why}")

    prev = {"sh.600036", "sh.600900", "sh.601601", "sz.000333", "sh.601398",
            "sh.600377", "sh.600018", "sh.600600", "sh.601088", "sz.000651"}
    now = set(sel.index)
    print(f"\n  CHANGE vs the previous ten: kept {len(prev & now)}, removed "
          f"{sorted(prev - now)}, added {sorted(now - prev)}")

    sel.to_parquet(OUT / "dividend_shortlist_v2.parquet")
    (OUT / "dividend_shortlist_v2.json").write_text(json.dumps(
        {"as_of": str(TODAY.date()), "fy_basis": 2025,
         "names": {c: {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v)
                       for k, v in r.items()} for c, r in sel.iterrows()},
         "removed": REMOVED, "watchlist": WATCHLIST, "excluded": EXCLUDED,
         "kept": sorted(prev & now), "added": sorted(now - prev),
         "dropped": sorted(prev - now)}, indent=2, default=float, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nwrote {OUT / 'dividend_shortlist_v2.parquet'}")


if __name__ == "__main__":
    main()
