# hermes-quant

A-share (大陆股市) quantitative research, backtesting, and paper-trading system.
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
            RQAlpha friction gate  (T+1 · 涨跌停 · 印花税 · 5元最低佣金 · 100股)
```

A staged pipeline; a strategy advances only when the prior stage holds up:

1. **Backtest** on historical data, offline. Every candidate must pass an
   A-share-faithful friction model (RQAlpha or vnpy.alpha) before advancing. vnpy's
   default CTA backtester is futures-style and overstates P&L at small accounts, where
   100-share lots, the 5元 minimum commission, 印花税, and T+1 dominate net returns;
   un-frictioned returns are not relied upon.
2. **Realtime paper trading** (模拟盘) at capital tiers grouped small/medium/large
   (1万·3万·5万 / 10万·50万 / 100万·500万). A monthly-rebalance strategy needs only an
   end-of-day feed, so paper trading is a lightweight idempotent EOD ledger that replays
   the same research engine forward (no train/serve skew); see docs/paper_trading.md.
   The tiers are configuration on one strategy object and make the small-account floor
   explicit: the book is infeasible below ~3万 (100-share lots + 5元 minimum commission).
3. **Live** (small real capital): deferred. The same strategy object, with the gateway swapped.

See [docs/architecture.md](docs/architecture.md) for the full stack rationale.

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
| **BaoStock** | none (anonymous) | free historical daily backbone — start here |
| **Tushare Pro** | free token (some fields need 积分) | financials, point-in-time index members, delisting |
| **AKShare** | none (scraper) | realtime L1 snapshot for paper trading only — fragile, not for the historical backbone |

Backtest window: **2015-01-01 → present** (multi-regime), with the most recent ~1–2 years
held out for walk-forward validation. **Delisted stocks are included** to avoid
survivorship bias. Price-limit rules differ by board/date (科创板/创业板 = ±20%).

## Layout

```
src/hermes/        importable package (src-layout, PyPA-recommended)
  paths.py         single source of truth for on-disk locations
  config.py        secret/token loading (env → .env.local)
  data/            ETL: vendor adapters → adjusted parquet data lake
  research/        offline: factors, backtest, eval (calibration metrics)
  live/            online EOD paper trading: strategy spec, feed, idempotent ledger
  execution/       vnpy strategy adapters (deferred: live broker gateway)
scripts/           runnable entrypoints (probes/ = one-off & superseded)
data/              local data lake — INPUTS (gitignored)
results/           generated OUTPUTS: signals, backtests, figures, models (gitignored)
external/          upstream framework checkouts, pip install -e (gitignored)
docs/              architecture & decisions (tracked: curated findings)
notebooks/         research scratch
```
