# scripts/probes/

One-off, exploratory, and superseded scripts kept for the record but NOT part of the active
research → paper → live pipeline. Nothing here is run on a schedule or imported by `hermes`.

- `smoke_baostock.py` — end-to-end BaoStock connectivity smoke test (single fixed name/date).
- `probe_baostock_pit.py` — early validation that BaoStock alone provides point-in-time HS300
  membership + delisted-name history (now productionized in `scripts/build_pit_dataset.py`).
- `probe_tushare.py`, `probe_tushare_fundamentals.py` — one-off checks of which Tushare Pro
  endpoints the token tier permits (the rate-limited tier is why the data layer runs on BaoStock).
- `probe_cb_freedata.py` — kill-first gate for the convertible-bond lake: can free sources
  (Eastmoney/Sina/JSL) reconstruct a survivorship-free universe, dead-bond bars, and the
  point-in-time conversion-price trail? Verdict and the pre-registered question: docs/cb_lake.md.
- `backtest_demo.py` — single-name double-MA friction demo (mechanism only, no edge); superseded
  by the portfolio-level research.
- `portfolio_demo.py` — non-PIT (current-membership) portfolio baseline; superseded by
  `scripts/survivorship_study.py`, the canonical survivorship-free study.
