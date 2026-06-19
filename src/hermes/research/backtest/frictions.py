"""A-share transaction cost model. These frictions DOMINATE net returns at small
accounts — the whole point of the friction gate is to make them explicit.

Retail defaults, mid-2026:
  - commission (佣金):    万2.5 (0.025%), 最低 5 元/笔
  - stamp tax (印花税):    万5 (0.05%), SELL side only (halved 2023-08)
  - transfer fee (过户费): 万0.1 (0.001%), both sides
  - slippage (滑点):       modeled as bps on the execution price, both sides
  - lot size (一手):       100 shares
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AShareCosts:
    commission_rate: float = 2.5e-4   # 佣金 万2.5
    min_commission: float = 5.0       # 每笔最低 5 元
    stamp_tax_sell: float = 5e-4      # 印花税 万5,仅卖出
    transfer_fee_rate: float = 1e-5   # 过户费 万0.1,双向
    slippage_bps: float = 5.0         # 滑点,单边,基点
    lot_size: int = 100               # 一手 = 100 股

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
