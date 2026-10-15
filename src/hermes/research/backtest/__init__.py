"""Backtest layer.

The hand-rolled A-share cost model lives here: 100-share lots, the ¥5 minimum commission,
stamp tax (sell-side ~0.05%) and slippage; T+1 holds by construction for a monthly book, and
price-limit no-fill is opt-in (`limit_block`, OFF by default).

An independent-engine friction gate (RQAlpha, installed editable in external/, or
vnpy.alpha's equity backtest) is planned, not in place: RQAlpha has been run once, on naive
momentum (docs/engine_validation.md), and the deployed strategy has not been through it.
NEVER trust vnpy's default CTA backtester P&L for small accounts — it is
a futures-style model and omits exactly the frictions that dominate ¥5k–¥30k returns.
"""
