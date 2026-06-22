"""EOD paper trading: run the DEPLOYED strategy forward on daily closes, recording every
fill in an idempotent ledger (live.ledger). At capital tiers (¥10k–¥5M) to expose the
small-account friction the research engine flags via avg_names_held.

Architecture (option A, monthly EOD): the research backtest engine IS the strategy brain.
`replay()` runs `signal_portfolio_backtest(..., collect_trades=True)` and folds its per-fill
trade log, day by day, into a LedgerState valued with the SAME `valuation_panel`. So paper P&L is
reconstructed from the seed by the exact research code -- no re-implementation, hence no
train/serve drift (the dominant silent alpha-killer). The ledger is seeded at PAPER_INCEPTION
(invested fully that day) and tracked forward, so `total_return` is the FORWARD paper record --
NOT the 2015-> backtest (which `live_step(inception=None)` reproduces, archived separately).

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
from ..io import atomic_to_parquet, atomic_write_text
from ..paths import PAPER_DIR, ensure_dirs
from ..research.backtest.frictions import AShareCosts
from ..research.backtest.portfolio import signal_portfolio_backtest, valuation_panel
from .ledger import LedgerState, fold_day
from .strategy import DEPLOYED, PAPER_INCEPTION, DeployedStrategy, deployed_signal


def replay(price: pd.DataFrame, signal: pd.DataFrame, seed_cash: float, *,
           n_hold: int = 10, costs: AShareCosts | None = None, members_asof=None,
           weight_asof=None, rebalance_band: int = 0,
           initial_rebalance: bool = False) -> tuple[LedgerState, object]:
    """Reconstruct the strategy's P&L as an idempotent ledger. Returns (ledger, result):
    `result` is the underlying PortfolioResult (for parity checks / stats); `ledger` is built
    by folding `result.trades` day by day from `seed_cash`, valued with `valuation_panel`.

    `initial_rebalance` invests the seed on the FIRST bar (paper inception); default off keeps the
    research backtest (and the parity tests) on the natural month-end schedule.

    The two equity series MUST agree (live.paper's only job is to record the engine's
    decisions, not re-decide) -- `scripts/paper_dryrun.py` asserts this as the anti-skew gate."""
    result = signal_portfolio_backtest(
        price, signal, seed_cash, n_hold=n_hold, costs=costs, members_asof=members_asof,
        weight_asof=weight_asof, rebalance_band=rebalance_band, collect_trades=True,
        initial_rebalance=initial_rebalance,
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
              costs: AShareCosts | None = None, persist: bool = True,
              inception: str | None = PAPER_INCEPTION) -> dict:
    """Run the DEPLOYED strategy as a PAPER account: seed `seed_cash` at the `inception` close,
    invest it fully into the current top-N there, and track forward to `as_of` (default: the latest
    bar on disk). Returns today's report. Idempotent (recompute-from-seed): re-running a date
    reproduces it, and forward-adjusted re-basing is absorbed (one consistent basis).

    The signal is computed over the FULL history (so factor lookbacks are satisfied), but the ledger
    is seeded only from `inception` -- so `total_return` / `max_drawdown` are the FORWARD paper record
    since inception, NOT the 2015-> backtest. Pass `inception=None` for the full-history backtest
    curve (archived separately). Reads the lake refreshed by live.feed; uses the SAME deployed_signal
    as research. Persists the equity curve, full trade log, and a JSON report under PAPER_DIR.
    """
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof_fn = membership_lookup(mdf)

    close = load_close_panel(codes=union, field="close")
    pe = load_close_panel(codes=union, field="peTTM")
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        close, pe = close.loc[close.index <= cutoff], pe.loc[pe.index <= cutoff]

    signal = deployed_signal(close, pe, asof_fn, spec)     # full-history signal (lookbacks satisfied)
    initial = False
    if inception is not None:
        incept = pd.Timestamp(inception)
        if (close.index >= incept).any():                  # seed the paper ledger AT inception,
            close = close.loc[close.index >= incept]        # invest fully on that bar, track forward
            signal = signal.loc[signal.index >= incept]
            initial = True
    ledger, res = replay(close, signal, seed_cash, n_hold=spec.n_hold, costs=costs,
                         members_asof=asof_fn, weight_asof=spec.weight_asof,
                         rebalance_band=spec.rebalance_band, initial_rebalance=initial)

    today = close.index[-1]
    today_fills = [{**t, "date": t["date"].strftime("%Y-%m-%d")} for t in res.trades
                   if t["date"] == today]
    # Observability: separate the DATA date (last bar) from the wall-clock RUN date, so a
    # holiday/weekend/pre-publication run (which idempotently re-computes the prior bar) is
    # distinguishable from a fresh trading-day update. `lake_lag_days` > a long weekend (~4d)
    # means the lake is stale (no new data ingested), surfaced as a banner by paper_live.
    run_dt = date.today()
    lake_lag = (run_dt - today.date()).days
    report = {
        "as_of": today.strftime("%Y-%m-%d"),          # last data bar (the strategy's clock)
        "run_date": run_dt.strftime("%Y-%m-%d"),       # wall-clock date this was computed
        "lake_lag_days": lake_lag,                     # run_date - as_of (calendar days)
        "fresh": lake_lag <= 4,                        # False => likely a stale/holiday no-op
        "seed_cash": seed_cash,
        "inception": close.index[0].strftime("%Y-%m-%d") if initial else None,  # forward record starts here
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
        atomic_to_parquet(ledger_equity(ledger).to_frame(), PAPER_DIR / f"curve_{tag}.parquet")
        atomic_to_parquet(pd.DataFrame(res.trades), PAPER_DIR / f"trades_{tag}.parquet")
        atomic_write_text(json.dumps(report, ensure_ascii=False, indent=2),
                          PAPER_DIR / f"report_{tag}.json")
    return report
