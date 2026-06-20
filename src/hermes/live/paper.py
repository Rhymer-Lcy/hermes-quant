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

import json
from datetime import date

import pandas as pd

from ..data.lake import load_close_panel
from ..data.membership import MEMBERSHIP_PARQUET, membership_lookup
from ..paths import PAPER_DIR, ensure_dirs
from ..research.backtest.frictions import AShareCosts
from ..research.backtest.portfolio import signal_portfolio_backtest, valuation_panel
from .ledger import LedgerState, fold_day
from .strategy import DEPLOYED, DeployedStrategy, deployed_signal


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


def live_step(seed_cash: float, as_of: str | None = None, *, spec: DeployedStrategy = DEPLOYED,
              costs: AShareCosts | None = None, persist: bool = True) -> dict:
    """Run the DEPLOYED strategy forward through `as_of` (default: latest bar on disk) on the
    CURRENT data lake, and return today's report. Idempotent: the ledger is recomputed from
    the immutable seed over all data each call (see replay) -- re-running a date reproduces it,
    and 前复权 re-basing is absorbed because the whole curve is recomputed on one basis.

    Reads the lake refreshed by live.feed; uses live.strategy.deployed_signal (the SAME spec
    research uses). Persists the equity curve, full trade log, and a JSON report under PAPER_DIR.
    """
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof_fn = membership_lookup(mdf)

    close = load_close_panel(codes=union, field="close")
    pe = load_close_panel(codes=union, field="peTTM")
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        close, pe = close.loc[close.index <= cutoff], pe.loc[pe.index <= cutoff]

    signal = deployed_signal(close, pe, asof_fn, spec)
    ledger, res = replay(close, signal, seed_cash, n_hold=spec.n_hold, costs=costs,
                         members_asof=asof_fn, weight_asof=spec.weight_asof,
                         rebalance_band=spec.rebalance_band)

    today = close.index[-1]
    today_fills = [{**t, "date": t["date"].strftime("%Y-%m-%d")} for t in res.trades
                   if t["date"] == today]
    report = {
        "as_of": today.strftime("%Y-%m-%d"),
        "seed_cash": seed_cash,
        "equity": float(res.equity.iloc[-1]),
        "total_return": float(res.total_return),
        "max_drawdown": float(res.max_drawdown),
        "n_positions": int(sum(1 for s in ledger.positions.values() if s > 0)),
        "avg_names_held": float(res.avg_names_held),
        "positions": dict(sorted(ledger.positions.items())),
        "today_trades": today_fills,
        "n_trades_total": len(res.trades),
    }
    if persist:
        ensure_dirs()
        tag = f"{int(seed_cash)}"
        ledger_equity(ledger).to_frame().to_parquet(PAPER_DIR / f"curve_{tag}.parquet")
        pd.DataFrame(res.trades).to_parquet(PAPER_DIR / f"trades_{tag}.parquet")
        (PAPER_DIR / f"report_{tag}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
