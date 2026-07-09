# Architecture & stack decisions

Captured from the evaluation that produced this project; review before changing the
stack, as several of the constraints are non-obvious.

## Selected stack: vnpy + Qlib, with changes

No single open-source project does research-grade A-share backtesting and retail
A-share live execution well, so a best-of-breed split is necessary. vnpy + Qlib is the
strongest pairing, though neither is usable as-is:

- **Qlib** — research/ML layer. A-share-native (trading calendar, point-in-time
  fundamentals, 25+ model zoo), still maintained (v0.9.7, Aug 2025; development energy has
  moved to the separate **RD-Agent** repo — pin a git SHA). Emits signals only (even
  "Online Serving" is prediction-only; next-day order generation is unsupported on public
  data), giving a clean handoff to an executor.
- **vnpy** — execution layer. Most actively maintained (v4.4.0, May 2026; ~42k stars) and
  the broadest open-source A-share **live** path; the only candidate spanning
  backtest→paper→live with one strategy callback. Its CTA backtester is futures-style and
  does not model stamp tax / T+1 / price-limit no-fill / 100-share lots / ¥5 minimum commission.
- **RQAlpha** — the friction gate. Models exactly those A-share frictions natively
  (verified v6.1.x). A mandatory pass before any strategy advances.

## Three required changes

1. **Gate every backtest through an A-share-faithful friction model** (RQAlpha or
   vnpy.alpha). vnpy's default backtester overstates P&L at ¥5k–¥30k accounts, where
   100-share lots, the ¥5 minimum commission, stamp tax, and T+1 dominate net returns.
2. **Start in the vnpy.alpha single-stack** for the first strategies. It reuses the same
   Alpha158 factor code from research to live, structurally eliminating the dominant silent
   failure mode, train/serve feature skew. Move to the full Qlib stack only when the
   deep-model zoo or RD-Agent is actually needed.
3. **Model size is not pursued for its own sake.** On low-SNR A-share data, deep models
   frequently lose to GBDT/linear out-of-sample, and DRL (FinRL/TradeMaster) carries a real
   sim-to-real gap at the small-account stage. Honest costs, point-in-time discipline, and
   out-of-sample survival matter more than model complexity. (As of writing: TradeMaster
   stale ~Feb 2025; AlphaGen near-abandoned ~Dec 2024.) Research, backtest, paper trading,
   and data ETL are CPU-bound and run locally; large-scale ML training (Qlib/RD-Agent
   sweeps, rolling retrain) is a deferred, unrealized option.

## Data

- **BaoStock** (free, anonymous, API) — the deployed pipeline's *sole* source: the historical
  daily backbone (incl. delisted names) and point-in-time HS300/CSI500 index membership.
- **Tushare Pro** (free token; optional extra, not installed by default) — **unused**. The
  adapter (`data/sources/tushare_source.py`) is retained only as a reference for a possible paid
  tier: its purpose was market cap for the size factor, but free-float cap now reconstructs from
  the BaoStock lake (`fl.float_cap`) and the size tilt was rejected outright (A4, risk_control.md).
- **AKShare** (free, scraper) — minute bars for the separate intraday futures line (`intraday/`)
  only; fragile (Sina scraping, with outage and IP-ban risk), never the backbone. EOD paper
  trading does *not* use it: `live/feed.py` refreshes from BaoStock.
- Backtest window **2015→present** (multi-regime); hold out the last ~1–2 years for
  walk-forward. **Include delisted stocks** (survivorship-bias guard). Handle per-board
  price limits (Main Board ±10%, STAR Market/ChiNext ±20%).
- Live feed (later): broker-backed miniQMT/xtdata is the only robust realtime path.

## Live & compliance (deferred — no real money yet)

- The miniQMT retail threshold is now commonly **~¥100k** at many brokers (not the old
  universal ¥500k), so a ¥100k paper→live step is realistically reachable.
- Per the 2025-07-07 SSE/SZSE/BSE programmatic-trading rules, **report the program to your
  broker before trading** (this applies to all programmatic trading, not only HFT; the book
  is far below the 300/sec, 20000/day HFT thresholds).
- Use the broker-routed QMT/Ptrade path; treat easytrader (GUI automation) as a disposable
  experiment, not infrastructure.

## Borrowed from odds-pipeline

- `paths.py` (single source of truth for directories) and `config.py` (secrets via
  env → `.env.local`).
- Offline (`historical/`) vs online (`live/`) separation.
- Calibration discipline (Brier / log loss / ECE / reliability) → `research/eval`.
- Model-vs-market divergence framing: alpha matters only where the model genuinely differs
  from what is priced.
- Idempotent ledger recomputed from a seed plus folded days → `live/ledger.py`.
- The governing principle from its README: an efficient market offers no systematic edge
  over the consensus; value is calibration, uncertainty identification, and sizing.
