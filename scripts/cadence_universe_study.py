"""Two untested axes the user flagged (data, not assertion):
  PART 1 -- rebalance CADENCE: does QUARTERLY match monthly at lower turnover/cost (helping the
            small tiers), and does WEEKLY just churn? Value is a slow factor, so a-priori Q~M, W worse.
  PART 2 -- COMBINED universe: does HS300 ∪ CSI500 beat either alone, or does adding the worse-beta
            small caps just dilute HS300? (A6 showed CSI500 alone is far worse.)

Deployed signal (value + 1m-reversal 5/1) throughout; PIT membership; A-share frictions; capped to
the common 2015-2025 window. CSI500/combined runs use price limit (daily ±10%/±20% limit) ON + ST filtered
(the small-cap-rigorous treatment); HS300-alone is the deployed baseline (no limits -- they don't bind for
large caps).

    python scripts/cadence_universe_study.py

NOTE: PART 2 (combined universe) needs the CSI500 dataset, which was purged (rejected; see
docs/risk_control.md A6/A7). Regenerate it first with `python scripts/build_csi500_dataset.py`
(coverage-gated; run until it reports complete); PART 1 (HS300 cadence) runs without it.
RE-VERIFIED 2026-07 on a gated full rebuild: PART 1 and PART 2 both reproduce their documented
rows within +-0.2pp CAGR, verdicts unchanged.
"""
from collections import Counter

import numpy as np
import pandas as pd

from hermes.data.ingest import BACKTEST_END
from hermes.data.lake import load_close_panel
from hermes.data.membership import (CSI500_MEMBERSHIP_PARQUET, MEMBERSHIP_PARQUET,
                                    membership_lookup)
from hermes.data.sources import baostock_source as bss
from hermes.research.backtest.limits import limit_flags
from hermes.research.backtest.portfolio import signal_portfolio_backtest as BT
from hermes.live.strategy import deployed_signal

END = BACKTEST_END        # single source of truth (data.ingest); do NOT re-hardcode


def cal(r):
    return r.cagr / abs(r.max_drawdown) if r.max_drawdown < 0 else float("nan")


def main() -> None:
    # ---- PART 1: cadence on HS300 (deployed value+rev) ----
    hs = pd.read_parquet(MEMBERSHIP_PARQUET); hs_union = sorted(hs["code"].unique())
    hs_asof = membership_lookup(hs)
    close = load_close_panel(codes=hs_union, field="close", end=END)
    pe = load_close_panel(codes=hs_union, field="peTTM", end=END)
    sig = deployed_signal(close, pe, hs_asof)

    print("PART 1 -- rebalance cadence (HS300 value+rev 5/1, top10, 2015-2025):")
    print(f"  {'tier':>9} {'freq':>5} {'CAGR':>7} {'maxDD':>7} {'Calmar':>7} {'costs':>11} {'rebals':>7}")
    for tier in (10_000, 30_000, 100_000, 500_000):
        for freq in ("Q", "M", "W"):
            r = BT(close, sig, tier, 10, members_asof=hs_asof, rebalance_freq=freq)
            print(f"  {tier:>9,} {freq:>5} {r.cagr:>+7.1%} {r.max_drawdown:>7.1%} {cal(r):>7.2f} "
                  f"{r.total_costs:>11,.0f} {r.n_rebalances:>7}")
        print()

    # ---- PART 2: combined HS300 ∪ CSI500 vs each alone (monthly) ----
    cs = pd.read_parquet(CSI500_MEMBERSHIP_PARQUET)
    comb = pd.concat([hs, cs]).drop_duplicates(["date", "code"]).reset_index(drop=True)
    comb_union = sorted(comb["code"].unique())
    cs_asof, comb_asof = membership_lookup(cs), membership_lookup(comb)
    print(f"PART 2 -- combined universe (HS300 657 ∪ CSI500 1326 = {len(comb_union)} names), value+rev 5/1, monthly:")

    cclose = load_close_panel(codes=comb_union, field="close", end=END)
    # DATA-COMPLETENESS GATE (same as csi500_universe_study): a partial CSI500 lake inverts
    # conclusions rather than merely degrading them; refuse to study one.
    have = int(cclose.notna().any().sum())
    if have / len(comb_union) < 0.99:
        raise RuntimeError(
            f"combined-universe data INCOMPLETE: {have}/{len(comb_union)} names have bars "
            f"({have / len(comb_union):.1%} < 99%). Rebuild with build_csi500_dataset.py first.")
    cpe = load_close_panel(codes=comb_union, field="peTTM", end=END)
    cpre = load_close_panel(codes=comb_union, field="preclose", end=END)
    cst = load_close_panel(codes=comb_union, field="isST", end=END)
    lb = limit_flags(cclose, cpre)
    not_st = ~(cst == True)
    with bss.session():
        ind = bss.stock_industry()
    code2ind = dict(zip(ind["code"], ind["industry"].replace("", np.nan)))

    def sig_for(asof, st_filter):
        c, p = (cclose.where(not_st), cpe.where(not_st)) if st_filter else (cclose, cpe)
        return deployed_signal(c, p, asof)

    def hhi(signal, asof, n):
        hh = []
        for d in [x for x in pd.to_datetime(sorted(comb["date"].unique())) if x in cclose.index]:
            row = signal.loc[d].dropna(); row = row[row.index.isin(asof(d))]
            top = row.sort_values(ascending=False).head(n).index
            if len(top):
                cnt = Counter(code2ind.get(c, "<none>") for c in top)
                hh.append(sum((v / len(top)) ** 2 for v in cnt.values()))
        return float(np.mean(hh)) if hh else float("nan")

    print(f"  {'universe':>22} {'nhold':>5} {'CAGR':>7} {'maxDD':>7} {'Calmar':>7} {'secHHI':>7}")
    runs = [
        ("HS300 alone (baseline)", hs_asof, False, None, 10),
        ("CSI500 alone", cs_asof, True, lb, 30),
        ("HS300∪CSI500 combined", comb_asof, True, lb, 10),
        ("HS300∪CSI500 combined", comb_asof, True, lb, 30),
    ]
    for tag, asof, stf, lim, n in runs:
        s = sig_for(asof, stf)
        r = BT(cclose, s, 1_000_000, n, members_asof=asof, limit_block=lim)
        print(f"  {tag:>22} {n:>5} {r.cagr:>+7.1%} {r.max_drawdown:>7.1%} {cal(r):>7.2f} {hhi(s, asof, n):>7.2f}")
    print("\nRead PART1: if Q's Calmar ~ M with lower costs/rebals, quarterly is the small-account win; "
          "W should churn (higher costs, no Calmar gain). PART2: combined beats HS300 only if Calmar rises "
          "vs the 0.32 baseline -- else adding small caps dilutes (expected from A6).")


if __name__ == "__main__":
    main()
