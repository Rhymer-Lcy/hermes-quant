"""Final long-horizon A-share dividend shortlist: assemble, add valuation context, verify.

The shortlist is CHOSEN, not ranked by a formula. Stage 1 and stage 2 are a funnel; the last step
is a judgement about business durability and portfolio construction that a composite score cannot
make -- in particular, 21 of the clean large-cap survivors are banks, and a mechanical top-10 would
be a single style bet wearing ten tickers.

Adds what A3 requires and the screens do not carry: each name's CURRENT valuation against its OWN
five-year history (PE and PB percentile from the local adjusted lake), plus a final re-verification
of every printed number.

    conda activate hermes
    python scripts/dividend_shortlist.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from hermes.data.lake import load_close_panel
from hermes.paths import RESULTS_DIR, ensure_dirs

OUT = RESULTS_DIR / "screens"
TODAY = pd.Timestamp("2026-09-21")
LOOKBACK = pd.Timestamp("2021-09-21")

# (code, bucket, one-line thesis) -- buckets are the three A3 categories.
SHORTLIST = [
    # --- durable dividend compounders: growing earnings AND growing DPS, payout with headroom ---
    ("sh.600036", "compounder", "retail-funded deposit franchise; best-in-class ROE among banks"),
    ("sh.600900", "compounder", "cascade hydro; near-annuity output, regulated, huge scale"),
    ("sh.601601", "compounder", "insurer with the lowest payout here -- dividend headroom"),
    ("sz.000333", "compounder", "global appliance platform; 20%+ ROE through a property downturn"),
    # --- stable high yield, low growth: the payout is the return, not the growth ---
    ("sh.601398", "stable", "mega-bank; policy-anchored payout, minimal growth"),
    ("sh.600377", "stable", "Jiangsu toll roads; concession cash, low capex intensity"),
    ("sh.600018", "stable", "Shanghai port throughput; low leverage, steady payout"),
    ("sh.600600", "stable", "beer premiumisation; staple demand, clean balance sheet"),
    # --- cyclical high dividend: real payers, but the payout tracks a commodity or a cycle ---
    ("sh.601088", "cyclical", "coal + captive rail/power; payout is high but cycle-linked"),
    ("sz.000651", "cyclical", "air-con leader; high yield and ROE, but DPS flat and H1 negative"),
]
# Deliberately EXCLUDED and why -- recorded so the omission is a decision, not an oversight.
EXCLUDED = {
    "sh.601006": "大秦铁路 -- earnings fell 119.3bn->59.0bn over two years and the DPS is already "
                 "cut 17.7% while the payout ratio rose to ~74%. A declining denominator.",
    "sh.600941": "中国移动 -- rejected by the screen (A-share listed 2022, so no six-year A-share "
                 "payout streak); separately, FY2025 profit turned negative and 2026H1 fell 6.3%.",
    "sh.601166": "兴业银行 -- 6.0% yield is attractive but four-year profit CAGR is -1.6% and H1 "
                 "-4.7%; kept off the list as a yield-without-growth case.",
    "sh.600329": "达仁堂 -- 13.6% printed yield is a one-off; OCF covers only 0.26x of it and H1 "
                 "profit fell 66.5%. The archetypal trap this screen exists to catch.",
}


def valuation_context(codes: list[str]) -> pd.DataFrame:
    pe = load_close_panel(codes=codes, field="peTTM")
    pb = load_close_panel(codes=codes, field="pbMRQ")
    rows = []
    for c in codes:
        r = {"code": c}
        for lab, panel in (("pe", pe), ("pb", pb)):
            if c not in panel.columns:
                r[f"{lab}_now"] = r[f"{lab}_pct5"] = np.nan
                continue
            s = panel[c].dropna()
            s = s[(s.index >= LOOKBACK) & (s.index <= TODAY)]
            s = s[s > 0]
            if s.empty:
                r[f"{lab}_now"] = r[f"{lab}_pct5"] = np.nan
                continue
            now = float(s.iloc[-1])
            r[f"{lab}_now"] = now
            r[f"{lab}_pct5"] = float((s < now).mean() * 100)
        rows.append(r)
    return pd.DataFrame(rows).set_index("code")


def main() -> None:
    ensure_dirs()
    s2 = pd.read_parquet(OUT / "dividend_stage2.parquet").set_index("code")
    codes = [c for c, _b, _t in SHORTLIST]
    missing = [c for c in codes if c not in s2.index]
    if missing:
        raise SystemExit(f"ABORT: shortlist names absent from the verified screen: {missing}")
    val = valuation_context(codes)

    rows = []
    for code, bucket, thesis in SHORTLIST:
        r = s2.loc[code]
        v = val.loc[code]
        rows.append({
            "code": code, "name": r["name"], "bucket": bucket,
            "industry": r["industry"], "price": r["price"], "yield": r["yield"],
            "dps_fy25": r["dps_latest"], "payout": r["payout_fy25"],
            "eps_cover": (r["eps_fy25"] / r["dps_latest"]) if r["dps_latest"] else np.nan,
            "ocf_cover": r["ocf_cover_fy25"],
            "roe5": r["roe_mean5"], "roe_min5": r["roe_min5"], "roe_std5": r["roe_std5"],
            "dps_cagr5": r["dps_cagr5"], "np_cagr4": r["np_cagr_4y"],
            "np_yoy_h1": r["np_yoy_h1"], "rev_yoy_h1": r["rev_yoy_h1"],
            "streak": r["streak"], "debt_ratio": r["debt_ratio"],
            "pe_now": v["pe_now"], "pe_pct5": v["pe_pct5"],
            "pb_now": v["pb_now"], "pb_pct5": v["pb_pct5"],
            "n_flags": int(r["n_flags"]), "thesis": thesis,
        })
    df = pd.DataFrame(rows).set_index("code")

    # --- final gates: nothing on this list may carry a trap flag or a broken streak ---
    bad = df[df["n_flags"] > 0]
    if len(bad):
        raise SystemExit(f"ABORT: flagged names on the shortlist: {list(bad.index)}")
    if int(df["streak"].min()) < 6:
        raise SystemExit("ABORT: a shortlist name has a payout streak under six years")
    banks = int((df["industry"].astype(str).str.contains("银行")).sum())
    print(f"shortlist: {len(df)} names, {df['bucket'].value_counts().to_dict()}, "
          f"banks = {banks}/{len(df)}")
    if banks > 3:
        raise SystemExit("ABORT: the shortlist is a disguised single-style bank bet")

    for bucket, label in (("compounder", "DURABLE DIVIDEND COMPOUNDERS"),
                          ("stable", "STABLE HIGH YIELD, LOW GROWTH"),
                          ("cyclical", "CYCLICAL HIGH DIVIDEND -- MORE CAUTION")):
        g = df[df["bucket"] == bucket]
        print(f"\n{'=' * 116}\n{label}\n{'=' * 116}")
        print(f"  {'code':>12} {'name':<8} {'industry':<10} {'px':>8} {'yld':>6} {'payout':>7} "
              f"{'EPSx':>5} {'OCFx':>6} {'ROE5':>6} {'dpsC5':>7} {'npC4':>7} {'H1np':>7} "
              f"{'PE':>6} {'PEpct':>6} {'PB':>5} {'PBpct':>6}")
        for c, r in g.iterrows():
            print(f"  {c:>12} {str(r['name'])[:8]:<8} {str(r['industry'])[:10]:<10} "
                  f"{r['price']:>8.2f} {r['yield']:>6.2%} {r['payout']:>7.1%} "
                  f"{r['eps_cover']:>5.2f} {r['ocf_cover']:>6.2f} {r['roe5']:>6.1%} "
                  f"{r['dps_cagr5']:>7.1%} {r['np_cagr4']:>7.1%} {r['np_yoy_h1']:>7.1f} "
                  f"{r['pe_now']:>6.1f} {r['pe_pct5']:>5.0f}% {r['pb_now']:>5.2f} "
                  f"{r['pb_pct5']:>5.0f}%")
        for c, r in g.iterrows():
            print(f"      {r['name']}: {r['thesis']}")

    print(f"\n{'=' * 116}\nPORTFOLIO-LEVEL READ\n{'=' * 116}")
    print(f"  equal-weight yield of the ten: {df['yield'].mean():.2%}")
    print(f"  median payout ratio          : {df['payout'].median():.1%}  "
          f"(range {df['payout'].min():.1%} - {df['payout'].max():.1%})")
    print(f"  names with OCF cover >= 1.0x : {int((df['ocf_cover'] >= 1).sum())}/{len(df)}")
    print(f"  names with 2026H1 profit down: {int((df['np_yoy_h1'] < 0).sum())}/{len(df)}")
    print(f"  industries represented       : {df['industry'].nunique()}")
    print(f"  cheapest vs own 5y PE        : "
          f"{df['pe_pct5'].idxmin()} at {df['pe_pct5'].min():.0f}th pct")
    print(f"  dearest vs own 5y PE         : "
          f"{df['pe_pct5'].idxmax()} at {df['pe_pct5'].max():.0f}th pct")

    print(f"\n{'=' * 116}\nDELIBERATELY EXCLUDED\n{'=' * 116}")
    for c, why in EXCLUDED.items():
        print(f"  {c}: {why}")

    df.to_parquet(OUT / "dividend_shortlist.parquet")
    (OUT / "dividend_shortlist.json").write_text(
        json.dumps({"as_of": str(TODAY.date()),
                    "names": {c: {k: (None if pd.isna(v) else v)
                                  for k, v in r.items()} for c, r in df.iterrows()},
                    "excluded": EXCLUDED}, indent=2, default=float, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nwrote {OUT / 'dividend_shortlist.parquet'}")


if __name__ == "__main__":
    main()
