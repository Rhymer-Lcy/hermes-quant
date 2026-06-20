# hermes-quant

A-share (大陆股市) quantitative research, backtesting, and paper-trading system.
Codename **Hermes** — the Greek god of commerce and trade.

> Status: research pipeline built and cross-validated — BaoStock historical pull →
> friction-faithful, point-in-time (survivorship-free) backtest → factor and walk-forward
> ML evaluation, cross-checked against RQAlpha (see docs/). The deployed strategy (value +
> a light 1-month-reversal tilt) and the EOD paper-trading ledger are built and run forward
> on real data (see docs/paper_trading.md). Next milestone: small live money via a broker
> gateway, after the paper-trading record holds up.

## Architecture

The system deliberately splits **offline research** from **online execution**,
because no single open-source tool does both well for A-shares.

```
            ┌─────────────────────────────┐         ┌──────────────────────────┐
            │  RESEARCH  (offline)         │ signals │  EXECUTION  (online)     │
            │  cluster: V100x8 / local PC  │ ──────▶ │  local PC (Windows)      │
            │                              │ (files) │                          │
            │  Qlib / vnpy.alpha           │         │  vnpy + paper account    │
            │  factors · models · backtest │         │  → (later) miniQMT live  │
            └─────────────────────────────┘         └──────────────────────────┘
                         │
                         ▼
            RQAlpha friction gate  (T+1 · 涨跌停 · 印花税 · 5元最低佣金 · 100股)
```

Three staged pipeline (a strategy only advances when the prior stage holds up):

1. **Backtest** on historical data — research/train offline; **every candidate must
   pass an A-share-faithful friction model** (RQAlpha or vnpy.alpha) before advancing.
   vnpy's default CTA backtester is futures-style and will *overstate* P&L at small
   accounts — never trust un-frictioned backtest returns.
2. **Realtime paper trading** (模拟盘) at capital tiers grouped small/medium/large
   (1万·5万 / 10万·50万 / 100万·500万). A monthly-rebalance strategy needs only an
   end-of-day feed, so paper trading is a lightweight idempotent EOD ledger that replays
   the SAME research engine forward (no train/serve skew); see docs/paper_trading.md.
   The tiers are a config on one strategy object — and expose that the book is infeasible
   below ~3万 (100-share lots + 5元 minimum commission).
3. **Live** (small real money) — *deferred*. Same strategy object, swap the gateway.

See [docs/architecture.md](docs/architecture.md) for the full stack rationale.

## Hardware split

| Workload | Where | GPU? |
|---|---|---|
| Factor/model training, HPO sweeps, rolling retrain | company V100×8 cluster | parallel jobs, one GPU per task |
| Backtest, paper trading, live execution, data ETL | local PC (i7-14700KF + RTX 5080) | none — these are CPU/process problems |

The cluster buys **research throughput and validation rigor**, not bigger models
or trading edge. A-share data is low signal-to-noise; honest costs, point-in-time
discipline, and out-of-sample survival matter more than model size.

## Environment

Conda env **`hermes`** (Python 3.12). Core research/data stack is installed.
The forks (vnpy etc.) are installed editable from [external/](external/README.md).

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

Backtest window: **2015-01-01 → present** (multi-regime), most recent ~1–2 years
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
external/          forks, pip install -e (gitignored)
docs/              architecture & decisions (tracked: curated findings)
notebooks/         research scratch
```
