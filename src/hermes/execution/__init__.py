"""Execution layer: vnpy strategy adapters for paper trading and (later) live.

The SAME strategy object runs across stages — only the gateway changes:
  paper:  vnpy_paperaccount (local matching against a live L1 snapshot feed)
  live:   vnpy_xt / miniQMT (deferred until a strategy is proven)

Factors consumed here must be the SAME code used in research, or scores must be
imported from the research artefact — recomputing factors independently is the
dominant silent alpha-killer (train/serve feature skew).
"""
