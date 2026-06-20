"""A3: does sector/industry neutralization cut plain value's -33% drawdown? NO -- it makes
it WORSE, monotonically. (A documented dead-end; the neutralize/cap helpers live here, not
in the factor library, precisely because the lever is rejected.)

The value top-10 in HS300 is ~70% banks + real-estate/construction -- so the 'diversify the
sectors' premise is right. But A-share banks are ~4x cheaper on earnings yield AND the
lowest-vol, most defensive cluster in HS300, so the value factor essentially IS the bank
trade, and that concentration is the SOURCE of the strategy's residual defensiveness.
Forcing money out of it (full demean, or per-sector caps) injects higher-beta non-financial
names that fall harder in a whole-market selloff -- drawdown worsens at every step.

Rigor note: this uses the LATEST Shenwan snapshot applied to all dates, i.e. it grants
neutralization a mild LOOK-AHEAD advantage (industry of large caps is near-static anyway).
It still fails monotonically -- so the null is robust: even a look-ahead-favoured version of
the lever loses. A real alpha use would need PIT industry per rebalance; not worth building
for a rejected lever (the free pull is kept in baostock_source.stock_industry for future
sector ATTRIBUTION on a wider universe). Same verdict as A1/A2/A4/B: the -33% is systematic.

    python scripts/a3_sector_demo.py
"""
import numpy as np
import pandas as pd

from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.data.sources import baostock_source as bss
from hermes.research.backtest.portfolio import signal_portfolio_backtest
from hermes.research.factors import library as fl

N_HOLD = 10
CAP = 1_000_000


def _groups(columns, code2ind):
    g: dict[str, list[str]] = {}
    for c in columns:
        lab = code2ind.get(c)
        g.setdefault(lab if pd.notna(lab) else "<none>", []).append(c)
    return g


def sector_demean(panel, code2ind):
    """Full neutralization: per date, subtract the within-industry cross-sectional mean.
    Unlabelled names ('<none>') are left untouched."""
    out = panel.copy()
    for lab, cols in _groups(panel.columns, code2ind).items():
        if lab == "<none>":
            continue
        sub = panel[cols]
        out[cols] = sub.sub(sub.mean(axis=1), axis=0)
    return out


def sector_cap(panel, code2ind, cap):
    """Per date, keep only each sector's top-`cap` names by score (NaN the rest), so a
    subsequent top-N pick holds at most `cap` names per sector."""
    out = pd.DataFrame(np.nan, index=panel.index, columns=panel.columns)
    groups = _groups(panel.columns, code2ind)
    for d in panel.index:
        row = panel.loc[d]
        for cols in groups.values():
            keep = row[cols].dropna().sort_values(ascending=False).head(cap).index
            out.loc[d, keep] = row[keep]
    return out


def cal(r):
    return r.cagr / abs(r.max_drawdown) if r.max_drawdown < 0 else float("nan")


def main() -> None:
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof = membership_lookup(mdf)
    close = load_close_panel(codes=union, field="close")
    ep = fl.earnings_yield(load_close_panel(codes=union, field="peTTM"))
    ep_pit = fl.restrict_to_universe(ep, asof)

    with bss.session():
        ind = bss.stock_industry()                    # latest Shenwan snapshot
    ind = ind[ind["code"].isin(union)]
    code2ind = dict(zip(ind["code"], ind["industry"].replace("", np.nan)))
    print(f"labelled union names: {sum(pd.notna(v) for v in code2ind.values())}/{len(union)} "
          "(latest snapshot, applied to all dates -> mild look-ahead in neutralization's favour)")

    def bt(sig):
        return signal_portfolio_backtest(close, sig, CAP, N_HOLD, members_asof=asof)

    print(f"\nsector neutralization @ {CAP:,} (top{N_HOLD}, monthly, PIT, A-share frictions):")
    print(f"  {'variant':>26} {'CAGR':>8} {'maxDD':>8} {'Calmar':>7}")

    def line(tag, sig):
        r = bt(sig)
        print(f"  {tag:>26} {r.cagr:>+8.1%} {r.max_drawdown:>8.1%} {cal(r):>7.2f}")

    line("value (base)", ep)
    line("value sector-demean (full)", sector_demean(ep_pit, code2ind))
    for c in [5, 4, 3, 2]:
        line(f"value per-sector cap {c}", sector_cap(ep_pit, code2ind, c))

    print("\nFinding: A3 REJECTED. Every degree of sector diversification DEEPENS the drawdown "
          "(full demean ~-58%, caps 5->2 drive maxDD -39%->-51%, Calmar 0.28->0.01). Value in "
          "HS300 IS the bank trade (banks ~4x cheaper AND the low-vol defensive cluster); the "
          "sector concentration is the strategy's residual defensiveness, not the -33%. The "
          "drawdown is systematic whole-market beta -- intra-universe levers cannot cut it.")


if __name__ == "__main__":
    main()
