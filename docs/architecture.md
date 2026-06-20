# Architecture & stack decisions

Captured from the evaluation that produced this project. Read this before changing the
stack — the constraints are non-obvious and load-bearing.

## Verdict: vnpy + Qlib, with changes

No single open-source project does research-grade A-share backtesting **and** retail
A-share live execution well, so a best-of-breed split is unavoidable. vnpy + Qlib is the
strongest pairing, but neither is turnkey:

- **Qlib** — research/ML layer. A-share-native (trading calendar, point-in-time
  fundamentals, 25+ model zoo), still maintained (v0.9.7, Aug 2025; development energy has
  moved to the separate **RD-Agent** repo — pin a git SHA). Emits signals only (even
  "Online Serving" is prediction-only; next-day order generation is unsupported on public
  data), giving a clean handoff to an executor.
- **vnpy** — execution layer. Most actively maintained (v4.4.0, May 2026; ~42k stars) and
  the broadest open-source A-share **live** path; the only candidate spanning
  backtest→paper→live with one strategy callback. Its CTA backtester is futures-style and
  does not model 印花税 / T+1 / 涨跌停 no-fill / 100股 / 5元最低佣金.
- **RQAlpha** — the friction gate. Models exactly those A-share frictions natively
  (verified v6.1.x). A mandatory pass before any strategy advances.

## Three non-optional changes

1. **Gate every backtest through an A-share-faithful friction model** (RQAlpha or
   vnpy.alpha). vnpy's default backtester overstates P&L at 5k–3万 accounts, where
   100-share lots, the 5元 minimum commission, 印花税, and T+1 dominate net returns.
2. **Start in the vnpy.alpha single-stack** for the first strategies. It reuses the same
   Alpha158 factor code from research to live, structurally eliminating the dominant silent
   failure mode, train/serve feature skew. Move to the full Qlib stack only when the
   deep-model zoo or RD-Agent is actually needed.
3. **Do not pursue model size for its own sake.** On low-SNR A-share data, deep models
   frequently lose to GBDT/linear out-of-sample, and DRL (FinRL/TradeMaster) carries a real
   sim-to-real gap at the small-account stage. Honest costs, point-in-time discipline, and
   out-of-sample survival matter more than model complexity. (As of writing: TradeMaster
   stale ~Feb 2025; AlphaGen near-abandoned ~Dec 2024.) Research, backtest, paper trading,
   and data ETL are CPU-bound and run locally; large-scale ML training (Qlib/RD-Agent
   sweeps, rolling retrain) is a deferred, unrealized option.

## Data

- **BaoStock** (free, anonymous, API) — historical daily backbone. Start here.
- **Tushare Pro** (free token; some fields need 积分) — financials, point-in-time index
  membership, delisting. Add when needed.
- **AKShare** (free, scraper) — realtime L1 snapshot for paper trading only; fragile
  (EastMoney/Sina scraping, with outage and IP-ban risk), never the backbone.
- Backtest window **2015→present** (multi-regime); hold out the last ~1–2 years for
  walk-forward. **Include delisted stocks** (survivorship-bias guard). Handle per-board
  price limits (主板 ±10%, 科创板/创业板 ±20%).
- Live feed (later): broker-backed miniQMT/xtdata is the only robust realtime path.

## Live & compliance (deferred — no real money yet)

- The miniQMT retail threshold is now commonly **~10万** at many brokers (not the old
  universal 50万), so a 10万 paper→live step is realistically reachable.
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
