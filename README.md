# hermes-quant

A-share (mainland China stock market) quantitative research, backtesting, and paper-trading system.
Codename **Hermes**.

> Status: the research pipeline is built and cross-validated — BaoStock historical
> pull → friction-faithful, point-in-time (survivorship-free) backtest → factor and
> walk-forward ML evaluation, cross-checked against RQAlpha (see docs/). The deployed
> strategy (value with a light 1-month-reversal tilt) and the end-of-day paper-trading
> ledger run forward on live data (see docs/paper_trading.md). Next milestone: small
> live capital through a broker gateway, once the paper-trading record holds up.

## Architecture

Offline research and online execution are separated deliberately, because no single
open-source tool does both well for A-shares.

```
            ┌─────────────────────────────┐         ┌──────────────────────────┐
            │  RESEARCH  (offline)         │ signals │  EXECUTION  (online)     │
            │  local PC (Windows)          │ ──────▶ │  local PC (Windows)      │
            │                              │ (files) │                          │
            │  Qlib / vnpy.alpha           │         │  vnpy + paper account    │
            │  factors · models · backtest │         │  → (later) miniQMT live  │
            └─────────────────────────────┘         └──────────────────────────┘
                         │
                         ▼
            RQAlpha friction gate  (T+1 · price limit · stamp tax · ¥5 minimum commission · 100-share lots)
```

A staged pipeline; a strategy advances only when the prior stage holds up:

1. **Backtest** on historical data, offline. Every candidate must pass an
   A-share-faithful friction model (RQAlpha or vnpy.alpha) before advancing. vnpy's
   default CTA backtester is futures-style and overstates P&L at small accounts, where
   100-share lots, the ¥5 minimum commission, stamp tax, and T+1 dominate net returns;
   un-frictioned returns are not relied upon.
2. **Realtime paper trading** (simulated account) at capital tiers grouped small/medium/large
   (¥10k·¥30k·¥50k / ¥100k·¥500k / ¥1M·¥5M). A monthly-rebalance strategy needs only an
   end-of-day feed, so paper trading is a lightweight idempotent EOD ledger that replays
   the same research engine forward (no train/serve skew); see docs/paper_trading.md.
   The tiers are configuration on one strategy object and make the small-account floor
   explicit: the book is infeasible below ~¥30k (100-share lots + ¥5 minimum commission).
   The paper ledger is a FORWARD record — seeded at an inception date and tracked from there
   (`PAPER_INCEPTION`), so its return is measured since inception, distinct from the 2015→
   backtest (reproduce that with `python scripts/paper_live.py --backtest`; archived under
   `results/backtests/`).
3. **Live** (small real capital): deferred. The same strategy object, with the gateway swapped.

See [docs/architecture.md](docs/architecture.md) for the full stack rationale.

**Built vs planned.** The research and paper-trading engine is hand-rolled in `src/hermes/`
and depends on no trading framework; the diagram above is the *target* stack. The external
frameworks are unmodified — **RQAlpha** is used only as an independent friction cross-check of
the hand-rolled backtest (see docs/engine_validation.md), while **vnpy** (execution) and
**Qlib** (ML research) are intended layers that are deferred and not yet used. Nothing under
`external/` is forked with local changes.

A-share data is low signal-to-noise; honest costs, point-in-time discipline, and
out-of-sample survival matter more than model size. Research, backtest, paper trading,
and data ETL are CPU-bound and run on a single local workstation.

## Environment

Conda env **`hermes`** (Python 3.12); the core research/data stack is installed. vnpy
and RQAlpha are installed editable from [external/](external/README.md) as pinned
upstream checkouts.

```
conda activate hermes
python scripts/probes/smoke_baostock.py    # verify the data link
```

## Data sources

| Source | Auth | Role |
|---|---|---|
| **BaoStock** | none (anonymous) | the deployed pipeline's sole source: historical daily backbone (incl. delisted names) **and** point-in-time HS300/CSI500 membership |
| **Tushare Pro** | free token (optional extra, not installed by default) | **unused** — adapter kept only as a reference for a possible paid tier; free-float cap now reconstructs from the BaoStock lake |
| **AKShare** | none (scraper) | minute bars for the separate intraday futures line only (`intraday/`) — fragile, never the backbone. Paper trading does **not** use it; it refreshes from BaoStock EOD |

Backtest window: **2015-01-01 → present** (multi-regime), with the most recent ~1–2 years
held out for walk-forward validation. **Delisted stocks are included** to avoid
survivorship bias. Price-limit rules differ by board/date (STAR Market/ChiNext = ±20%).

## Layout

```
src/hermes/        the engine — importable package (src-layout); no trading-framework dependency
  config.py        secret/token loading (env → .env.local)
  paths.py, io.py  on-disk locations; atomic file writes
  data/            vendor adapters (BaoStock; optional Tushare) → adjusted parquet lake; PIT HS300 membership
  research/
    backtest/      friction-faithful backtest engine: portfolio, frictions, limits, stops, hedge, sizing, regime
    factors/       factor library (value, reversal, low-vol, size, quality)
    eval/, model/  single-factor IC + calibration; walk-forward LightGBM combiner
  live/            EOD paper trading: strategy spec, data feed, idempotent ledger
  intraday/        separate intraday/futures research line (AKShare minute bars)
  execution/       vnpy live-gateway adapters — deferred stub, unused
scripts/           *_study.py = one research experiment, each written up in docs/ (risk_control A1–A9, no A5; multi_factor, factor_research, engine_validation, oos_decay); else operational drivers (paper_live, build_*, ingest_union)
  probes/          early one-off probes, superseded (kept for provenance)
tests/             pytest suite (95 tests): engine invariants, no-look-ahead, parity gates
data/              local data lake — INPUTS (gitignored)
results/           generated OUTPUTS: signals, backtests, figures, paper ledgers (gitignored)
external/          upstream checkouts (vnpy, RQAlpha), pip install -e — gitignored, unmodified
docs/              architecture & curated research findings (tracked)
notebooks/         research scratch
```
