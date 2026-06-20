"""Idempotent performance ledger — pattern borrowed from odds-pipeline build_review.

Principle: the running ledger is ALWAYS recomputed from an immutable seed plus
every folded day, so re-running a day is idempotent and the equity curve / trade
log is fully reproducible. Never mutate accumulated state in place.

The DECISIONS (which names, what target shares, fills) come from the SAME research
engine used in backtesting (see live.paper) -- the ledger only records and values them,
so paper trading cannot drift from research.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field


@dataclass
class LedgerState:
    """Recomputable ledger. `seed_cash` is immutable; everything else derives from
    folding daily records in chronological order."""
    seed_cash: float
    folded_days: list[str] = field(default_factory=list)  # YYYY-MM-DD, in order
    cash: float = 0.0
    positions: dict[str, int] = field(default_factory=dict)  # code -> shares
    equity_curve: list[tuple[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.equity_curve:
            self.cash = self.seed_cash


def fold_day(state: LedgerState, day: str, fills: list[dict], marks: dict[str, float]) -> LedgerState:
    """Apply one day's `fills` to a COPY of `state` and mark the book to `marks`, returning
    the new state (idempotent rebuilds call this for each day from the seed).

    `fills`: list of {code, shares (+buy/-sell), price (exec, incl. slippage), fee} -- the
    SAME records the backtest engine emits (PortfolioResult.trades). Cash moves by
    -(shares*price) - fee for every fill, so a buy debits and a sell credits uniformly.
    `marks`: code -> close price for valuation; a NaN/absent mark contributes 0 (matching
    the engine's _hold_value, which skips a name with no valid price that day).

    Folding the same `day` twice is a caller error (days must be distinct and chronological);
    the immutable-rebuild contract is: replay seed -> day_1 -> ... -> day_n reproduces state."""
    new = copy.deepcopy(state)
    for f in fills:
        code = f["code"]
        shares = int(f["shares"])
        new.cash += -(shares * f["price"]) - f["fee"]
        new.positions[code] = new.positions.get(code, 0) + shares
        if new.positions[code] == 0:
            del new.positions[code]
    equity = new.cash + sum(
        sh * marks[c] for c, sh in new.positions.items()
        if c in marks and not (isinstance(marks[c], float) and math.isnan(marks[c]))
    )
    new.folded_days.append(day)
    new.equity_curve.append((day, equity))
    return new
