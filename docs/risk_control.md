# Drawdown control (turning a signal into something deployable)

Plain value (earnings yield), survivorship-free PIT, top-10 equal weight: **+9.2% CAGR
but -33% max drawdown** -- not deployable as-is. This log records attempts to cut the
drawdown without giving back the return. The bar: raise **Calmar = CAGR / |maxDD|**.

## A1 — market-regime filter (沪深300 200-day MA): FAILED

Gate gross exposure to 0 when 沪深300 is below its 200-day MA, else 1 (decided monthly,
at the rebalance; PIT universe; A-share frictions).

| variant                    | CAGR  | maxDD  | Calmar |
|----------------------------|------:|-------:|-------:|
| value                      | +9.2% | -33.0% | 0.28   |
| value + 200d-MA regime     | +0.7% | -42.5% | 0.02   |

The filter HURT both axes: it gutted the return (9.2% -> 0.7%) and even worsened max
drawdown (-33% -> -42.5%). Why: A-share moves are sharp and V-shaped, so a lagging
200-day MA exits *after* the drop and re-enters *after* the recovery (whipsaw); the
monthly rebalance compounds the lag; and value's worst stretches are growth-led rallies
where the index is *above* its MA (risk-on) while cheap names keep falling -- exactly
when the filter stays fully invested. Trend-timing the index is not the tool here.

Reframe: a -33% drawdown is not extreme for long-only A-share equity (the index itself
saw large drawdowns over 2015-2025). The next lever is **position-level risk**
(volatility targeting, per-name and sector weight caps), not market timing. -> A2.

(The `exposure_asof` control added to the backtest is retained and reusable; it is the
filter *policy* -- a binary MA gate -- that failed, not the mechanism.)
