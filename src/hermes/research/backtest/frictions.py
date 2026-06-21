"""A-share transaction cost model. These frictions DOMINATE net returns at small
accounts — the whole point of the friction gate is to make them explicit.

Retail defaults, mid-2026:
  - commission:    2.5 bps (0.025%), minimum ¥5 per trade
  - stamp tax:     5 bps (0.05%), SELL side only (halved 2023-08)
  - transfer fee:  0.1 bps (0.001%), both sides
  - slippage:      modeled as bps on the execution price, both sides
  - lot size:      100 shares

The commission rate is the broker's ALL-IN quote, i.e. it already includes the exchange/
regulator handling and supervision fees (handling fee ~0.341 bps + securities supervision
fee ~0.2 bps ≈ 0.541 bps round-trip) that brokers collect bundled into commission -- so net
commission is ~1.96 bps + regulatory fees. We do NOT add the regulatory fees again on top
(double-counting). 2.5 bps all-in is a conservative retail level (many accounts negotiate
1.5-2 bps). NOT modeled: price-limit no-fill (rare for a monthly HS300 rebalance;
cross-checked in RQAlpha, which does model it) -- see docs/engine_validation.md.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AShareCosts:
    commission_rate: float = 2.5e-4   # commission, 2.5 bps
    min_commission: float = 5.0       # minimum ¥5 per trade
    stamp_tax_sell: float = 5e-4      # stamp tax, 5 bps, sell side only
    transfer_fee_rate: float = 1e-5   # transfer fee, 0.1 bps, both sides
    slippage_bps: float = 5.0         # slippage, one side, basis points
    lot_size: int = 100               # one lot = 100 shares

    def _commission(self, turnover: float) -> float:
        if turnover <= 0:
            return 0.0
        return max(turnover * self.commission_rate, self.min_commission)

    def buy_fees(self, turnover: float) -> float:
        """Cash fees on a buy of `turnover` RMB (slippage handled via exec price)."""
        return self._commission(turnover) + turnover * self.transfer_fee_rate

    def sell_fees(self, turnover: float) -> float:
        return self._commission(turnover) + turnover * (self.stamp_tax_sell + self.transfer_fee_rate)

    @property
    def slip(self) -> float:
        return self.slippage_bps * 1e-4


# All-zero costs — used to compute the "gross" (frictionless) curve for comparison.
ZERO_COSTS = AShareCosts(
    commission_rate=0.0, min_commission=0.0, stamp_tax_sell=0.0,
    transfer_fee_rate=0.0, slippage_bps=0.0,
)
