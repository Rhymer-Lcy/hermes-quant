"""Per-name stop-loss / take-profit exits (A9) -- an OPT-IN overlay on the monthly book.

The deployed book has no price stops. This module supplies the pure trigger logic so the ablation
in docs/risk_control.md A9 is reproducible; the engine (portfolio._score_backtest) consumes it via
the optional `stops=` argument and is bit-identical to the no-stop path when `stops is None`.

MODEL. A position's reference is its COST BASIS: the share-weighted average execution price actually
paid, slippage included (so the stop is measured against what the book paid, not a mid). An exit
liquidates the WHOLE position on that bar through the same `execute_orders` sell path as any other
sale -- slippage, commission, stamp tax all apply -- and the proceeds sit in cash until the next
scheduled rebalance, which may buy the name back if it still ranks. A stop is never checked on the
bar that opened the position (the engine evaluates exits before it rebalances).

TRIGGER MODES, and why the fills are deliberately pessimistic:
  - "close" (default): breach is judged on the close and the fill IS that close. Consistent with the
    engine, which executes everything at the close, and free of any intraday assumption. On a bar
    that pierces the stop and recovers, this does not exit at all; on a bar that gaps far through it,
    this exits far below the trigger. Neither error is systematically favourable.
  - "intraday": breach is judged on the bar's low/high, as a real resting order would be.
      * a STOP fills at min(stop_price, close) -- a stop is a MARKET order once touched, so a
        gap-through fills below the trigger, and the close is the only lower price the daily bar
        gives us. Never fills above the trigger.
      * a TAKE-PROFIT fills at exactly take_price -- a sell LIMIT fills AT the limit. It is never
        credited with the better price a gap-up would really have paid.
      * if BOTH are touched on one bar, the daily bar cannot say which came first, so the STOP is
        assumed to have fired. The worse outcome is always chosen.
    These conventions can only understate an overlay's benefit, never manufacture one -- the right
    bias for a lever we are trying to reject or accept honestly.

NOT MODELLED: intra-bar path, opening auctions, and price-limit no-fill (a name locked limit-down
cannot actually be sold; see research.backtest.limits). Turning limits on would make stops look
WORSE still, so leaving them off does not rescue a rejected overlay.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StopSpec:
    """`stop_loss`/`take_profit` are positive fractions of the cost basis (0.10 = -10% / +10%);
    None disables that leg. `trigger` is "close" or "intraday" (see the module docstring)."""

    stop_loss: float | None = None
    take_profit: float | None = None
    trigger: str = "close"

    def __post_init__(self) -> None:
        if self.trigger not in ("close", "intraday"):
            raise ValueError(f"trigger must be 'close' or 'intraday', got {self.trigger!r}")
        for name, v in (("stop_loss", self.stop_loss), ("take_profit", self.take_profit)):
            if v is not None and not 0.0 < v < 1.0:
                raise ValueError(f"{name} must be a fraction in (0, 1), got {v!r}")

    @property
    def active(self) -> bool:
        return self.stop_loss is not None or self.take_profit is not None


def exit_fill_price(spec: StopSpec, basis: float, close: float,
                    low: float = np.nan, high: float = np.nan) -> float | None:
    """Execution price if this bar exits a position carried at `basis`, else None.

    The returned price is the RAW fill (the engine still applies its sell-side slippage and fees).
    A non-tradable bar (NaN close) never exits: the book cannot sell what it cannot price."""
    if not spec.active or basis <= 0 or np.isnan(close):
        return None

    stop_px = basis * (1.0 - spec.stop_loss) if spec.stop_loss is not None else None
    take_px = basis * (1.0 + spec.take_profit) if spec.take_profit is not None else None

    if spec.trigger == "close":
        # stop_px < basis < take_px, so at most one of these can hold on a given close.
        if stop_px is not None and close <= stop_px:
            return close
        if take_px is not None and close >= take_px:
            return close
        return None

    hit_stop = stop_px is not None and not np.isnan(low) and low <= stop_px
    hit_take = take_px is not None and not np.isnan(high) and high >= take_px
    if hit_stop:                      # if both were touched, the stop is assumed to have fired first
        return min(stop_px, close)    # market order: a gap-through fills below the trigger
    if hit_take:
        return take_px                # sell limit: fills AT the limit, never above
    return None


def update_cost_basis(basis: dict[str, float], positions: dict[str, int],
                      fills: list[dict]) -> None:
    """Fold `execute_orders` fills into the share-weighted cost basis, in place.

    `positions` must already reflect the fills (execute_orders mutates it). A buy raises the basis
    toward its execution price; a partial sell leaves it unchanged (realised P&L does not re-price
    the remaining shares); a full exit drops the entry. Fills arrive sells-then-buys, one per code
    per side, so reconstructing the pre-buy share count as `positions[code] - qty` is exact."""
    for f in fills:
        code, qty, px = f["code"], f["shares"], f["price"]
        if qty > 0:
            prior = positions.get(code, 0) - qty
            if prior <= 0:
                basis[code] = px
            else:
                basis[code] = (basis.get(code, px) * prior + px * qty) / (prior + qty)
        elif code not in positions:
            basis.pop(code, None)
