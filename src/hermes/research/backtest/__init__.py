"""Backtest layer.

The mandatory A-share **friction gate** lives here: before any strategy advances
to paper trading, it must be evaluated under a faithful cost model —
T+1, 涨跌停 no-fill, 印花税 (sell-side ~0.05%), 最低5元佣金, 100-share lots, slippage.

Use RQAlpha (installed editable in external/) or vnpy.alpha's equity backtest as
the gate. NEVER trust vnpy's default CTA backtester P&L for small accounts — it is
a futures-style model and omits exactly the frictions that dominate 5k–3万 returns.
"""
