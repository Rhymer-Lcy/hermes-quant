"""Intraday / higher-frequency research line -- a SEPARATE domain from the monthly EOD equity
pipeline, isolated in its own top-level subpackage (own data adapters, engine, factors).

WHY isolated here (not a branch, not a separate repo): it shares the repo's tooling, conventions,
env, and verification discipline, but has a fundamentally different cadence (minute, not monthly),
instruments (T+0 futures / convertible bonds, not T+1 stocks), and microstructure -- so it gets its
own namespace and data, never touching the validated EOD engine. If it ever becomes an independent
trading system, fork it out then.

Scope decisions (see docs):
  - Instruments: 股指期货 IF/IC (CFFEX) -- T+0, and directly relevant to a future HEDGE OVERLAY that
    could cut the monthly book's systematic -33% drawdown; plus 商品期货 and 可转债 (both T+0).
  - Data: minute bars via AKShare (FREE, no token) -- futures via Sina (验证可用: futures_zh_minute_sina
    returns datetime/OHLC/volume/hold). CAVEAT: free minute history is SHALLOW (~recent window) and
    scraper-fragile; deep tick/L2 needs a paid vendor.
  - Engine: vnpy (already forked in external/) is the base for futures intraday backtesting -- it is
    built for minute/tick CTA, unlike the monthly EOD engine. Do NOT shoehorn the EOD engine here.
  - Mode: HISTORICAL RESEARCH + SIMULATION ONLY -- not live (live HF needs low-latency/colo/tick infra).
  - Method: start with simple, interpretable signals; neural nets only later, once a simple edge and a
    solid minute-data pipeline exist (intraday has the data volume to justify them; monthly does not).
"""
