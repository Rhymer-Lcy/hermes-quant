"""Eligibility-mask portfolio machinery for the friend's-ruleset studies (issues #2-#6).

Four of the five pre-registered rules reduce to the same primitive: a boolean (date x code)
mask saying which names the rule holds, rebalanced monthly into an equal-weight portfolio, or
a per-name target fraction traded against a held base. Costs are the repo's retail rates
(frictions.AShareCosts) applied PROPORTIONALLY to turnover; the two share-level effects that
cannot be expressed proportionally -- the 5 CNY minimum commission and 100-share lot rounding --
are excluded and disclosed in the writeups (at study scale they are sub-basis-point).

Execution convention throughout: signal at the close of day t, trade at the close of day t+1
(the repo's no-look-ahead rule). A NaN return inside a holding block (suspension) is treated
as 0: the name is held and untradeable, its mark stays put.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .frictions import AShareCosts


def proportional_rates(costs: AShareCosts = AShareCosts()) -> tuple[float, float]:
    """(buy, sell) cost per unit of turnover: commission + transfer + slippage, plus stamp tax
    on the sell side. The minimum-commission and lot-size terms are intentionally absent (see
    module docstring)."""
    buy = costs.commission_rate + costs.transfer_fee_rate + costs.slip
    sell = costs.commission_rate + costs.stamp_tax_sell + costs.transfer_fee_rate + costs.slip
    return buy, sell


def state_mask(enter: pd.DataFrame, exit_: pd.DataFrame, start_held: bool = False) -> pd.DataFrame:
    """Per-name persistent held state: True from an `enter` signal until an `exit_` signal.
    When both fire on the same day, exit wins (the conservative reading). Before any signal,
    the state is `start_held`."""
    sig = pd.DataFrame(np.nan, index=enter.index, columns=enter.columns)
    sig[enter.fillna(False)] = 1.0
    sig[exit_.fillna(False)] = 0.0
    return sig.ffill().fillna(1.0 if start_held else 0.0).astype(bool)


def month_end_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last trading day of each calendar month present in `idx`."""
    return pd.DatetimeIndex(idx.to_series().groupby(idx.to_period("M")).max().to_numpy())


def monthly_ew_backtest(eligible: pd.DataFrame, ret: pd.DataFrame,
                        costs: AShareCosts = AShareCosts()) -> dict:
    """Equal-weight portfolio of the names eligible at each month-end, executed at the next
    trading day's close, held (with intra-month drift) until the next execution.

    Returns dict with `net` and `gross` daily return Series (0 while fully in cash), the
    per-rebalance one-sided `turnover` Series, and the per-rebalance `n_held` Series.
    """
    idx = ret.index
    buy_rate, sell_rate = proportional_rates(costs)
    signals = month_end_dates(idx)
    positions = {}                        # signal date -> list of names
    for s in signals:
        row = eligible.loc[s]
        positions[s] = list(row.index[row.fillna(False)])

    gross = pd.Series(0.0, index=idx)
    net = pd.Series(0.0, index=idx)
    turnover_rows, held_rows = [], []
    prev_w = pd.Series(dtype=float)       # drifted weights at the end of the previous block

    ilocs = {d: i for i, d in enumerate(idx)}
    for k, s in enumerate(signals):
        e_i = ilocs[s] + 1                # execution: next trading day's close
        if e_i >= len(idx):
            break
        e = idx[e_i]
        end = idx[ilocs[signals[k + 1]] + 1] if k + 1 < len(signals) and \
            ilocs[signals[k + 1]] + 1 < len(idx) else idx[-1]
        names = positions[s]
        w0 = (pd.Series(1.0 / len(names), index=names) if names else pd.Series(dtype=float))

        # One-sided turnover at the execution close, priced buy/sell separately.
        alln = w0.index.union(prev_w.index)
        delta = w0.reindex(alln, fill_value=0.0) - prev_w.reindex(alln, fill_value=0.0)
        cost = delta.clip(lower=0.0).sum() * buy_rate + (-delta.clip(upper=0.0)).sum() * sell_rate
        turnover_rows.append((e, float(delta.abs().sum() / 2)))
        held_rows.append((e, len(names)))
        net[e] += -cost                   # the execution day itself earns no position return yet

        window = idx[(idx > e) & (idx <= end)]
        if len(window) and names:
            r = ret.loc[window, names].fillna(0.0)
            growth = (1.0 + r).cumprod()
            value = growth.mul(w0, axis=1).sum(axis=1)
            block = value / value.shift(1).fillna(1.0) - 1.0
            gross.loc[window] = block
            net.loc[window] += block
            prev_w = w0 * growth.iloc[-1] / value.iloc[-1]
        else:
            prev_w = pd.Series(dtype=float)

    return {"net": net, "gross": gross,
            "turnover": pd.Series(dict(turnover_rows), dtype=float),
            "n_held": pd.Series(dict(held_rows), dtype=float)}


def fractional_target_backtest(target: pd.DataFrame, ret: pd.DataFrame,
                               costs: AShareCosts = AShareCosts()) -> dict:
    """Per-name single-asset strategy that holds `target` fraction of account value in the
    stock (rest in cash at 0), trading ONLY at the close of a day whose target differs from
    the previous day's -- between signals the shares sit still and the fraction DRIFTS with
    the price, exactly like a real held position. `target` must already be shifted to
    execution timing by the caller (signal t -> row t+1). Day one starts at the first target
    with no entry cost (the buy-and-hold benchmark gets the identical free start, so the
    comparison is untouched).

    Returns dict of `net` and `gross` (date x code) return panels and the `traded` panel of
    absolute fraction changes.
    """
    buy_rate, sell_rate = proportional_rates(costs)
    tgt = target.to_numpy(dtype=float)
    r = np.nan_to_num(ret.to_numpy(dtype=float), nan=0.0)
    n_days, _ = r.shape
    gross = np.zeros_like(r)
    net = np.zeros_like(r)
    traded = np.zeros_like(r)
    frac = tgt[0].copy()
    for t in range(1, n_days):
        g = frac * r[t]
        gross[t] = g
        frac = frac * (1.0 + r[t]) / (1.0 + g)          # the held fraction drifts with price
        change = tgt[t] != tgt[t - 1]
        delta = np.where(change, tgt[t] - frac, 0.0)
        cost = np.clip(delta, 0.0, None) * buy_rate + np.clip(-delta, 0.0, None) * sell_rate
        net[t] = g - cost
        traded[t] = np.abs(delta)
        frac = np.where(change, tgt[t], frac)
    wrap = lambda a: pd.DataFrame(a, index=ret.index, columns=ret.columns)  # noqa: E731
    return {"net": wrap(net), "gross": wrap(gross), "traded": wrap(traded)}


def threshold_reversal_state(series: pd.Series, arm_pct: float, sell_pct: float,
                             lookback: int, reversal: int) -> pd.Series:
    """Issue #6's sentiment state machine on a single series: ARM when the level sits at or
    below its trailing `lookback`-day `arm_pct` quantile; BUY (state 1) on the first day, on
    or after arming, whose `reversal`-day change turns positive (arming resets after a buy);
    SELL (state 0) when the level reaches its trailing `sell_pct` quantile; otherwise the
    state persists. Starts flat, and stays flat until the rolling quantiles exist."""
    lo = series.rolling(lookback).quantile(arm_pct)
    hi = series.rolling(lookback).quantile(sell_pct)
    chg = series.diff(reversal)
    state, armed = 0.0, False
    out = np.zeros(len(series))
    for i in range(len(series)):
        if np.isfinite(lo.iloc[i]) and series.iloc[i] <= lo.iloc[i]:
            armed = True
        if armed and chg.iloc[i] > 0:
            state, armed = 1.0, False
        if np.isfinite(hi.iloc[i]) and series.iloc[i] >= hi.iloc[i]:
            state = 0.0
        out[i] = state
    return pd.Series(out, index=series.index)


def rolling_own_quantile(panel: pd.DataFrame, q: float, window: int = 1200,
                         min_obs: int = 750) -> pd.DataFrame:
    """Each name's trailing `window`-day `q`-quantile of its OWN history, NaN until `min_obs`
    observations have accrued (a thin window must disqualify the gate, not thin it)."""
    return panel.rolling(window, min_periods=min_obs).quantile(q)


def box_target(close: pd.DataFrame, window: int = 120, lo: float = 0.1, hi: float = 0.9,
               light: float = 0.5) -> pd.DataFrame:
    """Issue #5's box rule as a target-fraction panel (UNshifted -- signal timing): rolling
    min/max of the PRIOR `window` closes (today excluded); target `light` when the close is
    in the top (1-hi) slice of the box, 1.0 when in the bottom `lo` slice, previous target
    in between. Before the box exists the target is 1.0 (just hold)."""
    prior = close.shift(1)
    box_min = prior.rolling(window).min()
    box_max = prior.rolling(window).max()
    span = box_max - box_min
    upper = close >= box_min + hi * span
    lower = close <= box_min + lo * span
    tgt = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    tgt[lower] = 1.0
    tgt[upper & ~lower.fillna(False)] = light
    tgt[span.isna()] = 1.0
    return tgt.ffill().fillna(1.0)
