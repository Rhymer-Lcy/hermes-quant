"""CSI500 universe-expansion study: does going wider (1326 mid/small caps) cut plain value's
systematic ~-33% drawdown, and do QUALITY + factor diversification help here (unlike on HS300,
where they failed because value=banks)?

Rigorous for the small-cap universe: price-limit no-fill ON (limit_flags), ST names filtered out
(±5% limit + blow-up risk), A-share frictions, PIT CSI500 membership, full 2015-2025 window
(both universes capped to the same window for a fair head-to-head). All factors
restrict_to_universe'd to PIT members BEFORE blend (IRON RULE 1); n_hold swept (a wider universe
may want a wider basket); net AND gross Calmar; sector concentration measured.

    python scripts/csi500_universe_study.py

NOTE: CSI500 was REJECTED (see docs/risk_control.md A6), so its ~220MB local data was purged to
reclaim disk. This study is kept as the reproducible audit trail; to re-run it, first regenerate the
data with `python scripts/build_csi500_dataset.py` (free, ~minutes via BaoStock).
"""
from collections import Counter

import numpy as np
import pandas as pd

from hermes.data.ingest import BACKTEST_END
from hermes.data.lake import load_close_panel
from hermes.data.membership import (CSI500_MEMBERSHIP_PARQUET, MEMBERSHIP_PARQUET,
                                    membership_lookup)
from hermes.data.sources import baostock_source as bss
from hermes.paths import BACKTESTS_DIR
from hermes.research.backtest.frictions import ZERO_COSTS
from hermes.research.backtest.limits import limit_flags
from hermes.research.backtest.portfolio import signal_portfolio_backtest
from hermes.research.factors import library as fl

CAP = 1_000_000
END = BACKTEST_END        # single source of truth (data.ingest); do NOT re-hardcode


def cal(r):
    return r.cagr / abs(r.max_drawdown) if r.max_drawdown < 0 else float("nan")


def load_universe(parquet, end=END):
    mdf = pd.read_parquet(parquet)
    union = sorted(mdf["code"].unique())
    asof = membership_lookup(mdf)
    close = load_close_panel(codes=union, field="close", end=end)
    # DATA-COMPLETENESS GATE: the original A6 ran on a 97%-empty CSI500 pull (an unnoticed
    # mid-batch session-drop cascade) and published a conclusion computed from it. Refuse to
    # study a universe whose lake is materially incomplete.
    have = int(close.notna().any().sum())
    if have / len(union) < 0.99:
        raise RuntimeError(
            f"universe data INCOMPLETE for {parquet}: {have}/{len(union)} names have bars "
            f"({have / len(union):.1%} < 99%). Rebuild with build_csi500_dataset.py first.")
    pe = load_close_panel(codes=union, field="peTTM", end=end)
    pb = load_close_panel(codes=union, field="pbMRQ", end=end)
    pre = load_close_panel(codes=union, field="preclose", end=end)
    st = load_close_panel(codes=union, field="isST", end=end)
    evald = [d for d in pd.to_datetime(sorted(mdf["date"].unique())) if d in close.index]
    return mdf, union, asof, close, pe, pb, pre, st, evald


def main() -> None:
    print("loading CSI500 (1326) + HS300 (657) universes, capped to 2015-2025...")
    _, _, hs_asof, hs_close, hs_pe, _, _, _, _ = load_universe(MEMBERSHIP_PARQUET)
    _, c_union, c_asof, close, pe, pb, pre, st, evald = load_universe(CSI500_MEMBERSHIP_PARQUET)

    not_st = ~(st == True)                                  # ST -> excluded from selection
    def R(x):                                               # restrict-to-PIT + drop ST
        return fl.restrict_to_universe(x.where(not_st), c_asof)
    ep = R(fl.earnings_yield(pe)); q = R(fl.roe(pe, pb))
    rev = R(-fl.trailing_return(close, 20)); mom = R(fl.momentum(close, 120, 20))
    lowvol = R(fl.low_vol(close, 120))
    lb = limit_flags(close, pre)                            # price-limit no-fill panel (CSI500)

    with bss.session():
        ind = bss.stock_industry()
    code2ind = dict(zip(ind["code"], ind["industry"].replace("", np.nan)))

    def sector_hhi(sig, asof, n):
        hh = []
        for d in evald:
            if d not in sig.index:
                continue
            row = sig.loc[d].dropna(); row = row[row.index.isin(asof(d))]
            top = row.sort_values(ascending=False).head(n).index
            if len(top):
                cnt = Counter(code2ind.get(c, "<none>") for c in top)
                hh.append(sum((v / len(top)) ** 2 for v in cnt.values()))
        return float(np.mean(hh)) if hh else float("nan")

    def csi(sig, n, costs=None):
        return signal_portfolio_backtest(close, sig, CAP, n, costs=costs, members_asof=c_asof, limit_block=lb)

    # HS300 reference (same window, deployed config), limits OFF (don't bind for HS300).
    hs_ep = fl.restrict_to_universe(fl.earnings_yield(hs_pe), hs_asof)
    hs_rev = fl.restrict_to_universe(-fl.trailing_return(hs_close, 20), hs_asof)
    hs = signal_portfolio_backtest(hs_close, fl.blend([hs_ep, hs_rev], [5, 1]), CAP, 10, members_asof=hs_asof)

    print(f"\n2015-2025 @ {CAP:,}; CSI500 with price limit ON + ST filtered; NET/GROSS Calmar, maxDD, secHHI:")
    print(f"  {'config':>30} {'CAGR':>7} {'maxDD':>7} {'netCal':>7} {'grsCal':>7} {'secHHI':>7}")
    print(f"  {'HS300 value+rev 5/1 (deployed)':>30} {hs.cagr:>+7.1%} {hs.max_drawdown:>7.1%} "
          f"{cal(hs):>7.2f} {'--':>7} {'--':>7}")
    rows = []

    def line(tag, sig, n):
        rn, rg = csi(sig, n), csi(sig, n, ZERO_COSTS)
        print(f"  {tag:>30} {rn.cagr:>+7.1%} {rn.max_drawdown:>7.1%} {cal(rn):>7.2f} {cal(rg):>7.2f} "
              f"{sector_hhi(sig, c_asof, n):>7.2f}")
        rows.append({"config": tag, "cagr": rn.cagr, "max_dd": rn.max_drawdown,
                     "net_calmar": cal(rn), "gross_calmar": cal(rg)})

    for n in (20, 30, 50):
        line(f"CSI500 value (ep) n{n}", ep, n)
    line("CSI500 value+rev 5/1 n20", fl.blend([ep, rev], [5, 1]), 20)
    line("CSI500 value+quality 2/1 n20", fl.blend([ep, q], [2, 1]), 20)
    line("CSI500 val+qual+rev 3/2/1 n20", fl.blend([ep, q, rev], [3, 2, 1]), 20)
    line("CSI500 val+qual+rev 3/2/1 n30", fl.blend([ep, q, rev], [3, 2, 1]), 30)
    line("CSI500 5-factor eq n30", fl.blend([ep, q, rev, mom, lowvol], [1, 1, 1, 1, 1]), 30)
    pd.DataFrame(rows).to_csv(BACKTESTS_DIR / "csi500_universe_study.csv", index=False)
    print("\nKey questions: does CSI500 (wider, more sectors) CUT the ~-33% maxDD vs HS300, and do "
          "quality/diversification HELP here (vs hurting on HS300)? Read maxDD + secHHI + net/gross Calmar.")


if __name__ == "__main__":
    main()
