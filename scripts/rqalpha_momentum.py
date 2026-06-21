"""RQAlpha cross-check of the survivorship-free momentum result.

Runs the SAME strategy -- monthly equal-weight top-N HS300 by trailing return -- on
RQAlpha's battle-tested A-share engine + RiceQuant's own price data, but driven by
OUR BaoStock point-in-time HS300 membership (the free bundle's native index_components
needs paid rqdatac). RQAlpha natively handles dividends/splits, price-limit no-fill, T+1,
stamp tax, 100-share lots, and per-trade minimum commission -- so this is an INDEPENDENT
engine + price-data + corporate-action validation of the hand-rolled PIT backtest
(which produced ~ -7%/yr, survivorship-free), holding the membership definition fixed.

Run with the rqalpha env:
    D:\\Anaconda3\\envs\\rqalpha\\python.exe scripts/rqalpha_momentum.py
"""
import bisect

import pandas as pd
from rqalpha import run_func

# RQAlpha injects the strategy API (history_bars, order_target_percent, scheduler,
# context.portfolio/now, ...) into init/rebalance globals at run time -- NOT imported.
# noqa: F821

N_HOLD = 10
LOOKBACK = 20
MEMBERSHIP_PATH = r"f:\hermes-quant\data\parquet\hs300_membership.parquet"


def _to_rq(bao_code: str) -> str:
    """'sh.600000' -> '600000.XSHG'; 'sz.000001' -> '000001.XSHE'."""
    mkt, num = bao_code.split(".")
    return f"{num}.{'XSHG' if mkt == 'sh' else 'XSHE'}"


def init(context):
    mdf = pd.read_parquet(MEMBERSHIP_PATH)
    mdf["rq"] = mdf["code"].map(_to_rq)
    context.snaps = {d: set(g["rq"]) for d, g in mdf.groupby("date")}
    context.snap_dates = sorted(context.snaps)
    scheduler.run_monthly(rebalance, tradingday=1)  # noqa: F821


def _members_asof(context, when) -> set:
    ts = pd.Timestamp(when)
    i = bisect.bisect_right(context.snap_dates, ts) - 1
    return context.snaps[context.snap_dates[i]] if i >= 0 else set()


def rebalance(context, bar_dict):
    members = _members_asof(context, context.now)
    rets = {}
    for s in members:
        bars = history_bars(s, LOOKBACK + 1, "1d", "close")  # noqa: F821
        if bars is not None and len(bars) >= LOOKBACK + 1 and bars[0] > 0:
            rets[s] = bars[-1] / bars[0] - 1.0
    top = sorted(rets, key=rets.get, reverse=True)[:N_HOLD]

    for s in list(context.portfolio.positions.keys()):
        if s not in top and context.portfolio.positions[s].quantity > 0:
            order_target_percent(s, 0)  # noqa: F821
    weight = 1.0 / N_HOLD
    for s in top:
        order_target_percent(s, weight)  # noqa: F821


def run_one(capital: float) -> dict:
    config = {
        "base": {
            "start_date": "2015-01-01",
            "end_date": "2025-12-31",
            "frequency": "1d",
            "accounts": {"stock": capital},
        },
        "mod": {"sys_analyser": {"enabled": True, "plot": False}},
    }
    result = run_func(init=init, handle_bar=lambda c, b: None, config=config)
    return result["sys_analyser"]["summary"]


def main() -> None:
    print(f"{'capital':>12} {'CAGR':>9} {'total':>11} {'maxDD':>8} {'sharpe':>8}")
    for cap in [100_000, 1_000_000]:
        s = run_one(cap)
        print(f"{cap:>12,} {s.get('annualized_returns', float('nan')):>+8.1%} "
              f"{s.get('total_returns', float('nan')):>+10.1%} "
              f"{s.get('max_drawdown', float('nan')):>7.1%} "
              f"{s.get('sharpe', float('nan')):>8.2f}")


if __name__ == "__main__":
    main()
