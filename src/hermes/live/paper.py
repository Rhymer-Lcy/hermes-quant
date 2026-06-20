"""EOD paper trading: run the DEPLOYED strategy forward on daily closes, recording every
fill in an idempotent ledger (live.ledger). At capital tiers (5千–50万) to expose the
small-account friction the research engine flags via avg_names_held.

Architecture (option A, monthly EOD): the research backtest engine IS the strategy brain.
`replay()` runs `signal_portfolio_backtest(..., collect_trades=True)` over the data available
so far and folds its per-fill trade log, day by day, into a LedgerState valued with the SAME
`valuation_panel`. So paper P&L is reconstructed from the immutable seed by the exact research
code -- no re-implementation, hence no train/serve drift (the dominant silent alpha-killer).

Going live forward is then one step (live_step): append today's real EOD bar to the close
panel, recompute scores with the SAME factor code, re-run replay -- the ledger extends by the
new day(s). vnpy_paperaccount / vnpy_xt (intraday, realtime) stay deferred for higher-frequency
or true-live use; a monthly rebalance does not need them.
"""
from __future__ import annotations

import pandas as pd

from ..research.backtest.frictions import AShareCosts
from ..research.backtest.portfolio import signal_portfolio_backtest, valuation_panel
from .ledger import LedgerState, fold_day


def replay(price: pd.DataFrame, signal: pd.DataFrame, seed_cash: float, *,
           n_hold: int = 10, costs: AShareCosts | None = None, members_asof=None,
           weight_asof=None, rebalance_band: int = 0) -> tuple[LedgerState, object]:
    """Reconstruct the strategy's P&L as an idempotent ledger. Returns (ledger, result):
    `result` is the underlying PortfolioResult (for parity checks / stats); `ledger` is built
    by folding `result.trades` day by day from `seed_cash`, valued with `valuation_panel`.

    The two equity series MUST agree (live.paper's only job is to record the engine's
    decisions, not re-decide) -- `scripts/paper_demo.py` asserts this as the anti-skew gate."""
    result = signal_portfolio_backtest(
        price, signal, seed_cash, n_hold=n_hold, costs=costs, members_asof=members_asof,
        weight_asof=weight_asof, rebalance_band=rebalance_band, collect_trades=True,
    )
    valuation, _, _ = valuation_panel(price)

    fills_by_day: dict[pd.Timestamp, list[dict]] = {}
    for t in result.trades:
        fills_by_day.setdefault(t["date"], []).append(t)

    state = LedgerState(seed_cash=seed_cash)
    for d in valuation.index:
        marks = valuation.loc[d].to_dict()
        state = fold_day(state, d.strftime("%Y-%m-%d"), fills_by_day.get(d, []), marks)
    return state, result


def ledger_equity(state: LedgerState) -> pd.Series:
    """The ledger's equity curve as a date-indexed Series (for plotting / comparison)."""
    idx = pd.to_datetime([d for d, _ in state.equity_curve])
    return pd.Series([v for _, v in state.equity_curve], index=idx, name="equity")
