"""Double-low portfolio accounting for the pre-registered CB study (docs/cb_lake.md).

Small and bespoke on purpose -- the equity engine's frictions (T+1, 100-share lots, stamp
tax) do not apply to convertible bonds (T+0, no stamp tax, 10-bond lots negligible at
study scale). Conventions, all frozen in docs/cb_lake.md before any result existed:

  - signal at each month-end close, EXECUTION at the next trading day's close -- the
    version a retail replicator can actually trade (index_effect_study precedent);
  - equal weight across the n_hold LOWEST double-low scores; proportional cost per side
    on traded value;
  - suspended on the execution day: a pending SALE waits for the next tradable close; a
    pending BUY is dropped for the period (you cannot buy a halted bond), its slice stays
    in cash; a held name re-selected while suspended simply stays held;
  - a name whose series has ENDED exits at its last close -- the terminal-value
    convention, since the delisted-segment tail is not in free data -- or at ZERO when in
    `zero_mark` (the credit-default stress variant).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CBBacktestResult:
    equity: pd.Series          # net equity at each close from the first execution day (starts 1.0)
    rebalances: pd.DataFrame   # per rebalance: exec_date, n_selected, oneway_turnover

    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] / self.equity.iloc[0] - 1.0)

    @property
    def cagr(self) -> float:
        years = (self.equity.index[-1] - self.equity.index[0]).days / 365.25
        return float((self.equity.iloc[-1] / self.equity.iloc[0]) ** (1.0 / years) - 1.0)

    @property
    def max_drawdown(self) -> float:
        eq = self.equity
        return float((eq / eq.cummax() - 1.0).min())


def double_low_backtest(close: pd.DataFrame, score: pd.DataFrame, n_hold: int | None,
                        cost_per_side: float,
                        zero_mark: frozenset[str] = frozenset()) -> CBBacktestResult:
    """Run the frozen double-low design.

    close: daily panel (DatetimeIndex x code), the TRADABLE record -- NaN where the bond
        did not trade that day (pre-listing, suspension, post-delisting).
    score: rows indexed by SIGNAL dates (must be members of close.index, each with a next
        trading day), columns as in close; NaN = ineligible at that rebalance. The n_hold
        smallest scores are held equal-weight until the next rebalance; n_hold=None holds
        ALL eligible names (the equal-weight universe benchmark; pass cost_per_side=0).
    zero_mark: codes whose post-death value is zero instead of the last close.
    """
    if not score.index.isin(close.index).all():
        raise ValueError("every signal date must be a member of close.index")
    codes = list(close.columns)
    dates = close.index
    tradable = close.to_numpy(dtype=float)                     # NaN = cannot trade
    path = close.ffill().to_numpy(dtype=float)                 # mark-to-last-close
    last_pos = np.array([-1 if (lv := close[c].last_valid_index()) is None
                         else dates.get_loc(lv) for c in codes])
    for j, c in enumerate(codes):                              # stress mark: dead => worthless
        if c in zero_mark and last_pos[j] + 1 < len(dates):
            path[last_pos[j] + 1:, j] = 0.0

    col = {c: j for j, c in enumerate(codes)}
    exec_of = {}                                               # exec position -> selection (j list)
    for sig in score.index:
        pos = dates.get_loc(sig)
        if pos + 1 >= len(dates):
            continue                                           # no next trading day yet
        row = score.loc[sig].dropna()
        row = row.nsmallest(n_hold) if n_hold is not None else row
        exec_of[pos + 1] = [col[c] for c in row.index]

    if not exec_of:
        raise ValueError("no executable rebalance inside the panel")
    start = min(exec_of)
    units = np.zeros(len(codes))
    pending = np.zeros(len(codes), dtype=bool)
    cash = 1.0
    eq, rebs = [], []

    for i in range(start, len(dates)):
        for j in np.flatnonzero(pending):                      # deferred sales
            price = tradable[i, j] if np.isfinite(tradable[i, j]) else (
                path[i, j] if i > last_pos[j] else np.nan)     # dead => the terminal convention
            if np.isfinite(price):
                cash += units[j] * price * (1.0 - cost_per_side)
                units[j], pending[j] = 0.0, False

        if i in exec_of:
            held = np.flatnonzero(units)
            value = cash + (float(np.nansum(units[held] * path[i, held])) if len(held) else 0.0)
            selected = exec_of[i]
            keep = set(selected)
            traded = 0.0
            for j in held:                                     # exits first
                if j in keep:
                    continue
                if np.isfinite(tradable[i, j]) or i > last_pos[j]:
                    price = tradable[i, j] if np.isfinite(tradable[i, j]) else path[i, j]
                    proceeds = units[j] * price
                    cash += proceeds * (1.0 - cost_per_side)
                    traded += proceeds
                    units[j] = 0.0
                else:
                    pending[j] = True                          # suspended but alive: sell later
            if selected:
                # A deferred sale is not investable cash yet: size targets off what can
                # actually be deployed, or a big suspended exit would imply leverage.
                pend_val = float(np.nansum(units[pending] * path[i, pending])) if pending.any() else 0.0
                target = (value - pend_val) / len(selected)
                for j in selected:
                    pending[j] = False                         # re-selected: cancel a deferred sale
                    if not np.isfinite(tradable[i, j]):
                        continue                               # halted: keep if held, else stay cash
                    delta = target - units[j] * tradable[i, j]
                    cash -= delta + cost_per_side * abs(delta)
                    units[j] = target / tradable[i, j]
                    traded += abs(delta)
            rebs.append((dates[i], len(selected), traded / (2.0 * value) if value else 0.0))

        held = np.flatnonzero(units)
        eq.append(cash + (float(np.nansum(units[held] * path[i, held])) if len(held) else 0.0))

    # Prepend 1.0 at the signal close so the first entry's cost shows up in total_return.
    equity = pd.Series([1.0] + eq, index=dates[start - 1:], name="equity")
    rebalances = pd.DataFrame(rebs, columns=["exec_date", "n_selected", "oneway_turnover"])
    return CBBacktestResult(equity=equity, rebalances=rebalances)
