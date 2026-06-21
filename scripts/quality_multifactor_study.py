"""Does adding QUALITY (ROE) + factor diversification improve the deployed HS300 monthly
strategy, and does it cure the ~70%-banks sector concentration? Measure before adopting.

For each factor mix: net & gross Calmar, maxDD, AND top-10 sector concentration (mean HHI +
mean max single-sector share, via the free Shenwan industry snapshot). A real improvement must
either lift Calmar robustly (gross too) or cut sector concentration WITHOUT giving back Calmar.
ROE is reconstructed free from the lake (pbMRQ/peTTM); all factors restrict_to_universe'd to PIT
members BEFORE blend (IRON RULE 1). Full curve, not a single point (IRON RULE 2).

    python scripts/quality_multifactor_study.py
"""
from collections import Counter

import numpy as np
import pandas as pd

from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.data.sources import baostock_source as bss
from hermes.paths import BACKTESTS_DIR
from hermes.research.backtest.frictions import ZERO_COSTS
from hermes.research.backtest.portfolio import signal_portfolio_backtest
from hermes.research.factors import library as fl

N_HOLD = 10
CAP = 1_000_000


def main() -> None:
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof = membership_lookup(mdf)
    close = load_close_panel(codes=union, field="close")
    pe = load_close_panel(codes=union, field="peTTM")
    pb = load_close_panel(codes=union, field="pbMRQ")
    ps = load_close_panel(codes=union, field="psTTM")

    ep = fl.restrict_to_universe(fl.earnings_yield(pe), asof)
    q = fl.restrict_to_universe(fl.roe(pe, pb), asof)
    rev = fl.restrict_to_universe(-fl.trailing_return(close, 20), asof)
    mom = fl.restrict_to_universe(fl.momentum(close, 120, 20), asof)
    lowvol = fl.restrict_to_universe(fl.low_vol(close, 120), asof)

    with bss.session():
        ind = bss.stock_industry()
    ind = ind[ind["code"].isin(union)]
    code2ind = dict(zip(ind["code"], ind["industry"].replace("", np.nan)))
    eval_dates = [d for d in pd.to_datetime(sorted(mdf["date"].unique())) if d in close.index]

    def cal(r):
        return r.cagr / abs(r.max_drawdown) if r.max_drawdown < 0 else float("nan")

    def bt(sig, costs=None):
        return signal_portfolio_backtest(close, sig, CAP, N_HOLD, costs=costs, members_asof=asof)

    def sector_conc(sig):
        """Mean top-10 sector HHI and mean max single-sector share over the rebalance dates."""
        hhis, maxs = [], []
        for d in eval_dates:
            if d not in sig.index:
                continue
            row = sig.loc[d].dropna()
            row = row[row.index.isin(asof(d))]
            top = row.sort_values(ascending=False).head(N_HOLD).index
            if len(top) == 0:
                continue
            cnt = Counter(code2ind.get(c, "<none>") for c in top)
            shares = [v / len(top) for v in cnt.values()]
            hhis.append(sum(s * s for s in shares))
            maxs.append(max(shares))
        return float(np.mean(hhis)), float(np.mean(maxs))

    configs = {
        "value (base)": ep,
        "value+rev 5/1 (DEPLOYED)": fl.blend([ep, rev], [5, 1]),
        "value+quality 2/1": fl.blend([ep, q], [2, 1]),
        "value+quality 1/1": fl.blend([ep, q], [1, 1]),
        "val+qual+rev 4/2/1": fl.blend([ep, q, rev], [4, 2, 1]),
        "val+qual+rev 3/2/1": fl.blend([ep, q, rev], [3, 2, 1]),
        "5-factor (eq)": fl.blend([ep, q, rev, mom, lowvol], [1, 1, 1, 1, 1]),
        "val+qual+rev+mom 3/2/1/1": fl.blend([ep, q, rev, mom], [3, 2, 1, 1]),
    }

    rows = []
    print(f"HS300 top{N_HOLD} monthly @ {CAP:,}; NET/GROSS Calmar, maxDD, sector concentration:")
    print(f"  {'config':>26} {'CAGR':>7} {'maxDD':>7} {'netCal':>7} {'grsCal':>7} {'secHHI':>7} {'maxSec%':>7}")
    for tag, sig in configs.items():
        rn, rg = bt(sig), bt(sig, ZERO_COSTS)
        hhi, mx = sector_conc(sig)
        print(f"  {tag:>26} {rn.cagr:>+7.1%} {rn.max_drawdown:>7.1%} {cal(rn):>7.2f} {cal(rg):>7.2f} "
              f"{hhi:>7.2f} {mx:>7.0%}")
        rows.append({"config": tag, "cagr": rn.cagr, "max_dd": rn.max_drawdown, "net_calmar": cal(rn),
                     "gross_calmar": cal(rg), "sector_hhi": hhi, "max_sector_share": mx})
    pd.DataFrame(rows).to_csv(BACKTESTS_DIR / "quality_multifactor.csv", index=False)
    print("\nRead: a config is worth adopting only if it lifts net Calmar robustly (gross too) OR cuts "
          "sector concentration (HHI / maxSec%) WITHOUT giving back Calmar. secHHI ~1/n_eff: lower = "
          "more sectors. Baseline value is ~banks; see if quality/diversification spreads it.")


if __name__ == "__main__":
    main()
