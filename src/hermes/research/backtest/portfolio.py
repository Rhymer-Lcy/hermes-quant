"""Cross-sectional portfolio backtest with A-share frictions.

Monthly rebalance to an equal-weight top-N basket by a ranking SCORE. Long-only.
No-lookahead: the score is read at the month-end (signal date) and executed at the
NEXT trading day's close. T+1 is respected (monthly holding). 100-share lots,
per-trade ¥5 minimum commission, stamp tax, and slippage are all modeled.

Two entry points share one engine (`_score_backtest`):
  - momentum_portfolio_backtest: score = trailing `lookback`-day return.
  - signal_portfolio_backtest:   score = an arbitrary external signal panel
    (e.g. a walk-forward ML model's out-of-sample predictions).

PURPOSE: expose the small-account problem (`avg_names_held` = EFFECTIVE
diversification vs target `n_hold`) and run survivorship-free PIT studies.

Delisting/removal: a holding whose price series permanently ends is force-liquidated
once at its last real price net of fees and is never valued past that bar -- otherwise
a dead name would re-enter P&L at a stale forward-filled price (survivorship bias).

DOCUMENTED SIMPLIFICATIONS (not hidden): (1) price-limit no-fill is not modeled here (rare
for a monthly rebalance on liquid HS300); see the RQAlpha cross-check. (2) Lot-sizing
uses the forward-adjusted price LEVEL (approximate at the smallest tiers; cross-check confirms it
is second-order). Friction/feasibility tool, not validated alpha.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .frictions import AShareCosts


@dataclass
class PortfolioResult:
    equity: pd.Series
    total_return: float
    cagr: float
    max_drawdown: float
    n_rebalances: int
    target_n_hold: int
    avg_names_held: float      # EFFECTIVE diversification (mean held names/day)
    total_costs: float
    trades: list[dict] = field(default_factory=list)   # audit log if collect_trades=True:
    #   {date, code, shares (+buy/-sell), price (exec, incl. slippage), fee}


def _max_drawdown(eq: np.ndarray) -> float:
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def valuation_panel(price: pd.DataFrame):
    """Mark-to-market panel + delisting bookkeeping, shared by the backtest engine and the
    paper ledger so both value the book identically. Returns (valuation, last_valid,
    last_price): `valuation` carries the last price through INTERIOR halts but is NaN after
    each name's final real bar (an unbounded ffill would value a delisted name forever);
    `last_valid[code]`/`last_price[code]` are that final bar's date/price (for forced exit)."""
    price = price.sort_index()
    last_valid = {c: price[c].last_valid_index() for c in price.columns}
    last_price = {c: price[c].loc[lv] for c, lv in last_valid.items() if lv is not None}
    valuation = price.ffill()
    for c, lv in last_valid.items():
        if lv is not None:
            valuation.loc[valuation.index > lv, c] = np.nan
    return valuation, last_valid, last_price


def _hold_value(positions: dict[str, int], val: pd.Series) -> float:
    tot = 0.0
    for code, sh in positions.items():
        p = val.get(code, 0.0)
        if not np.isnan(p):
            tot += sh * p
    return tot


def _select_top(ranked: list[str], held: set[str], n_hold: int, band: int) -> list[str]:
    """Top-`n_hold` by score, with a turnover buffer (hysteresis): an existing holding is
    KEPT while it stays within the top `n_hold + band` -- it need not re-enter the strict
    top-`n_hold` -- so names hovering at the cutoff don't churn in and out every month. New
    names must rank in the top `n_hold` to enter. band=0 reduces to a plain top-`n_hold`.
    Critical for high-turnover signals (e.g. short-term reversal) where round-trip frictions
    can eat the factor's edge. Precondition: `ranked` is unique (the engine passes a unique
    column index); a non-unique `ranked` could duplicate a held name."""
    if band <= 0:
        return ranked[:n_hold]
    chosen = [c for c in ranked[:n_hold + band] if c in held][:n_hold]   # keep incumbents in exit zone
    for c in ranked[:n_hold]:                                            # fill rest with fresh entrants
        if len(chosen) >= n_hold:
            break
        if c not in chosen:
            chosen.append(c)
    return chosen


# --- shared rebalance primitives ----------------------------------------------------
# These three functions are the SINGLE source of truth for "turn a ranked basket into
# orders". Both the backtest engine (_score_backtest) and the paper-trading ledger
# (live.paper) call them, so paper trading cannot silently drift from research (the
# dominant train/serve alpha-killer). Keep them pure and free of any backtest-loop state.

def basket_weights(top: list[str], weight_asof, when) -> dict[str, float]:
    """Intra-basket weights over `top`, summing to 1. Equal weight if `weight_asof` is
    None (or yields a non-positive sum); otherwise the normalized non-negative weights
    from weight_asof(when, top) (e.g. inverse-vol)."""
    if weight_asof is None:
        return {c: 1.0 / len(top) for c in top}
    raw_w = weight_asof(when, top)
    s = sum(max(raw_w.get(c, 0.0), 0.0) for c in top)
    if s <= 0:
        return {c: 1.0 / len(top) for c in top}
    return {c: max(raw_w.get(c, 0.0), 0.0) / s for c in top}


def target_shares(gross: float, weights: dict[str, float], scale: float,
                  raw: pd.Series, slip: float, lot: int) -> dict[str, int]:
    """Lot-rounded target share count per name. `scale` = len(top)/n_hold keeps the
    gross-invested fraction identical to equal weight (unfilled slots stay cash). A name
    with no tradable price that day gets no target (left untouched by the caller)."""
    desired: dict[str, int] = {}
    for code, wt in weights.items():
        p = raw.get(code, np.nan)
        if not np.isnan(p):
            target_val = gross * wt * scale
            desired[code] = int(target_val // (p * (1 + slip) * lot)) * lot
    return desired


def execute_orders(cash: float, positions: dict[str, int], desired: dict[str, int],
                   raw: pd.Series, costs: AShareCosts, slip: float, lot: int,
                   block_buy: set | None = None, block_sell: set | None = None) -> tuple[float, float, list[dict]]:
    """Move `positions` toward `desired` shares: sells first (free up cash), then buys
    (capped by available cash, with a one-lot retry if fees tip it over). Mutates
    `positions` in place; returns (new_cash, cost_delta, fills). `cost_delta` is the sum of
    slippage + fees; each fill is {code, shares (+buy/-sell), price (exec, incl. slippage), fee}.

    `block_buy`/`block_sell`: codes that CANNOT be bought / sold on the exec day (price-limit no-fill
    -- a name locked limit-up has no sellers so a buy can't fill; locked limit-down, no buyers so
    a sell can't fill). Both default to empty (no blocking) so the HS300 deployed path is unchanged;
    populated only for the wider/small-cap universe where limits bind (see research.backtest.limits)."""
    block_buy = block_buy or set()
    block_sell = block_sell or set()
    cost_delta = 0.0
    fills: list[dict] = []
    for code, tgt in desired.items():                  # sells first
        cur = positions.get(code, 0)
        if tgt >= cur or code in block_sell:           # limit-down: no buyers, can't sell
            continue
        p = raw.get(code, np.nan)
        if np.isnan(p):
            continue
        qty = cur - tgt
        ep = p * (1 - slip)
        turnover = qty * ep
        fee = costs.sell_fees(turnover)
        cash += turnover - fee
        cost_delta += (qty * p - turnover) + fee
        positions[code] = cur - qty
        if positions[code] == 0:
            del positions[code]
        fills.append({"code": code, "shares": -qty, "price": ep, "fee": fee})
    for code, tgt in desired.items():                  # then buys (capped by cash)
        cur = positions.get(code, 0)
        if tgt <= cur or code in block_buy:            # limit-up: no sellers, can't buy
            continue
        p = raw.get(code, np.nan)
        if np.isnan(p):
            continue
        ep = p * (1 + slip)
        affordable = int(cash // (ep * lot)) * lot
        qty = min(tgt - cur, max(affordable, 0))
        if qty <= 0:
            continue
        turnover = qty * ep
        fee = costs.buy_fees(turnover)
        if turnover + fee > cash:
            qty -= lot
            if qty <= 0:
                continue
            turnover = qty * ep
            fee = costs.buy_fees(turnover)
        cash -= turnover + fee
        cost_delta += (turnover - qty * p) + fee
        positions[code] = cur + qty
        fills.append({"code": code, "shares": qty, "price": ep, "fee": fee})
    return cash, cost_delta, fills


def _score_backtest(price: pd.DataFrame, scores: pd.DataFrame, capital: float,
                    n_hold: int, costs: AShareCosts | None, members_asof,
                    exposure_asof=None, weight_asof=None, rebalance_band: int = 0,
                    collect_trades: bool = False, limit_block: pd.DataFrame | None = None,
                    rebalance_freq: str = "M") -> PortfolioResult:
    """Engine: each month hold the top-`n_hold` names by `scores` (read at the month-end
    signal date, executed next trading day), with A-share frictions. Weighting is equal
    by default; `weight_asof` supplies an alternative intra-basket weighting (e.g.
    inverse-vol) WITHOUT changing the gross-invested fraction, so a weighting scheme is
    compared to equal weight on like terms. `rebalance_band` adds a turnover buffer (see
    _select_top): incumbents are kept while within the top n_hold+band, cutting churn."""
    costs = costs or AShareCosts()
    lot = costs.lot_size
    slip = costs.slip

    price = price.sort_index()
    valuation, last_valid, last_price = valuation_panel(price)
    dates = price.index
    n = len(dates)

    periods = dates.to_period(rebalance_freq)          # "M" monthly (default), "Q" quarterly, "W" weekly
    pos_of = {d: i for i, d in enumerate(dates)}
    period_end = pd.Series(dates, index=dates).groupby(periods).max().tolist()
    rebal_exec = {pos_of[sig] + 1: pos_of[sig] for sig in period_end if pos_of[sig] + 1 < n}

    cash = float(capital)
    positions: dict[str, int] = {}
    total_costs = 0.0
    names_held_daily = []
    equity = np.empty(n)
    trades: list[dict] = []

    for i in range(n):
        di = dates[i]
        raw = price.iloc[i]                    # raw price = tradability + exec price
        val = valuation.iloc[i]

        # Force-exit holdings whose data has permanently ended (delisting / removal +
        # terminal halt): liquidate once at the last real price net of fees.
        for code in list(positions.keys()):
            lv = last_valid.get(code)
            if lv is not None and di > lv:
                qty = positions.pop(code)
                turnover = qty * last_price[code]
                fee = costs.sell_fees(turnover)
                cash += turnover - fee
                total_costs += fee
                if collect_trades:
                    trades.append({"date": di, "code": code, "shares": -qty,
                                   "price": last_price[code], "fee": fee})

        if i in rebal_exec:
            sd = dates[rebal_exec[i]]          # signal (month-end) date
            if sd in scores.index:
                f = scores.loc[sd].dropna()
                f = f[raw.reindex(f.index).notna()]            # tradable at exec
                if members_asof is not None:                   # point-in-time universe
                    f = f[f.index.isin(members_asof(sd))]
                ranked = f.sort_values(ascending=False).index.tolist()
                held = {c for c, sh in positions.items() if sh > 0}
                top = _select_top(ranked, held, n_hold, rebalance_band)

                if top:
                    equity_now = cash + _hold_value(positions, val)
                    exposure = exposure_asof(sd) if exposure_asof is not None else 1.0
                    gross = equity_now * exposure
                    # `scale` keeps the gross-invested fraction identical to equal weight
                    # (unfilled target slots stay cash), so a weighting scheme is compared
                    # on like terms -- only the split WITHIN the basket changes.
                    w = basket_weights(top, weight_asof, sd)
                    desired = target_shares(gross, w, len(top) / n_hold, raw, slip, lot)
                    for code in list(positions.keys()):            # sell anything dropped from top
                        desired.setdefault(code, 0)
                    block_buy = block_sell = None
                    if limit_block is not None and di in limit_block.index:
                        fl = limit_block.loc[di]                    # price-limit no-fill on the EXEC day
                        block_buy = set(fl.index[fl == 1])
                        block_sell = set(fl.index[fl == -1])
                    cash, cost_delta, fills = execute_orders(cash, positions, desired, raw, costs,
                                                             slip, lot, block_buy, block_sell)
                    total_costs += cost_delta
                    if collect_trades:
                        trades.extend({**fl, "date": di} for fl in fills)

        equity[i] = cash + _hold_value(positions, val)
        names_held_daily.append(sum(1 for sh in positions.values() if sh > 0))

    years = max((dates[-1] - dates[0]).days / 365.25, 1e-9)
    return PortfolioResult(
        equity=pd.Series(equity, index=dates),
        total_return=float(equity[-1] / capital - 1.0),
        cagr=float((equity[-1] / capital) ** (1.0 / years) - 1.0),
        max_drawdown=_max_drawdown(equity),
        n_rebalances=len(rebal_exec),
        target_n_hold=n_hold,
        avg_names_held=float(np.mean(names_held_daily)),
        total_costs=float(total_costs),
        trades=trades,
    )


def momentum_portfolio_backtest(panel: pd.DataFrame, capital: float, n_hold: int = 10,
                                lookback: int = 20, costs: AShareCosts | None = None,
                                members_asof=None) -> PortfolioResult:
    """Top-N by trailing `lookback`-day return. `members_asof`: optional
    callable(signal_date)->set[code] for the point-in-time universe."""
    scores = panel / panel.shift(lookback) - 1.0
    return _score_backtest(panel, scores, capital, n_hold, costs, members_asof)


def signal_portfolio_backtest(price: pd.DataFrame, signal: pd.DataFrame, capital: float,
                              n_hold: int = 10, costs: AShareCosts | None = None,
                              members_asof=None, exposure_asof=None,
                              weight_asof=None, rebalance_band: int = 0,
                              collect_trades: bool = False, limit_block: pd.DataFrame | None = None,
                              rebalance_freq: str = "M") -> PortfolioResult:
    """Top-N by an external `signal` panel (date x code), e.g. walk-forward ML
    out-of-sample predictions. `price` is the forward-adjusted close panel for exec/valuation.
    `exposure_asof`: optional callable(signal_date)->float in [0,1] scaling gross
    exposure (e.g. a market-regime filter); the remainder is held as cash.
    `weight_asof`: optional callable(signal_date, codes)->{code: weight} for intra-basket
    weighting (e.g. inverse-vol); equal weight if omitted.
    `rebalance_band`: turnover buffer (keep incumbents within top n_hold+band); 0 = off.
    `collect_trades`: also return the per-fill audit log (consumed by live.paper).
    `limit_block`: optional (date x code) price-limit flag panel (research.backtest.limits.limit_flags);
    blocks buys at the up-limit / sells at the down-limit on the exec day. OFF (None) by default
    -- liquid HS300 doesn't need it; supply it for the CSI500/small-cap universe."""
    return _score_backtest(price, signal, capital, n_hold, costs, members_asof,
                           exposure_asof, weight_asof, rebalance_band, collect_trades, limit_block,
                           rebalance_freq)
