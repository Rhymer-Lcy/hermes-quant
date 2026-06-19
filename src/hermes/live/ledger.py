"""Idempotent performance ledger — pattern borrowed from odds-pipeline build_review.

Principle: the running ledger is ALWAYS recomputed from an immutable seed plus
every folded day, so re-running a day is idempotent and the equity curve / trade
log is fully reproducible. Never mutate accumulated state in place.

SKELETON — fill in fold_day() once the paper-trading loop produces daily fills.
"""
from __future__ import annotations

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


def fold_day(state: LedgerState, day: str, fills: list[dict]) -> LedgerState:
    """Apply one day's fills to a COPY of state and return it (idempotent rebuilds
    call this for each day from the seed). NOT YET IMPLEMENTED."""
    raise NotImplementedError("fold_day: implement once paper-trading fills exist")
