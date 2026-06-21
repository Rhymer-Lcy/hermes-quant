# Paper trading (the second workflow stage: backtest → **paper** → small live)

The research stage closed with a deployable strategy whose risk is understood (value + a light
1-month-reversal tilt, net Calmar ~0.32, maxDD −33% systematic; see
[multi_factor.md](multi_factor.md) / [risk_control.md](risk_control.md)). Paper trading runs
that exact strategy *forward* on fresh end-of-day data, at capital tiers, before any real money.

## Architecture — option A: lightweight EOD ledger

A monthly-rebalance strategy does not require a tick/realtime gateway. The chosen design:

> The research backtest engine is the strategy logic; paper trading only records its decisions.

`live.paper.replay()` runs `signal_portfolio_backtest(..., collect_trades=True)` over the data
available so far and folds its per-fill trade log, day by day, into an idempotent `LedgerState`
(`live.ledger.fold_day`), valued with the same `valuation_panel` the engine uses. Paper P&L
is therefore the research engine's P&L reconstructed from an immutable seed: there is no second
implementation of factors, selection, sizing, or valuation, and hence no train/serve skew (the
dominant silent source of alpha decay). `scripts/paper_demo.py` asserts that the ledger equity equals the
engine equity bar-for-bar at every tier; if that gate fails, paper has drifted from research.

Going forward is one daily step (`live.paper.live_step`): refresh the lake to today, recompute
scores with the same factor code, and replay, extending the ledger by the new day(s).

### One source of truth for the strategy

The deployed spec lives in exactly one place, `live/strategy.py` (`DEPLOYED` +
`deployed_signal`), imported by both the research demo and the live driver. Neither
re-spells the 5:1 value/reversal blend, so they cannot diverge. `test_paper.py` locks
`deployed_signal` to the documented blend.

## Forward-only rigor (things the backtest never had to face)

1. **Forward-adjusted re-basing is not append-only.** Forward-adjusted prices rescale the entire
   history whenever a dividend/split occurs, so naively appending new days would mix two
   adjustment bases in one series. `live.feed.update_daily_bars` therefore performs a full
   re-pull (overwrite) of the union over `[BACKTEST_START, today]`, putting the whole lake on
   one consistent basis; `replay` then recomputes the ledger wholesale from the seed, so the
   equity curve is always self-consistent and re-running a date reproduces it. Forward-adjusted
   prices reinvest dividends via the adjustment, so the paper curve approximates a total-return account.
   *Deferred refinement:* explicit corporate-action cash/tax accounting (dividend cash timing,
   dividend tax) — second-order for a monthly large-cap book, required before live.

2. **Suspension vs delisting at the right edge.** The engine force-liquidates a holding once its
   price series permanently ends (NaN after `last_valid_index`). Forward, a name suspended for
   the last few days is indistinguishable from a delisting and may be liquidated early. For HS300 large
   caps, multi-day suspensions are rare and the effect is conservative (cash sits idle until the
   next rebalance, never overstating return). A membership-aware rule (liquidate only when also
   dropped from the index) is a documented future refinement.

3. **Membership must stay current.** `live.feed.extend_membership` pulls HS300 month-end
   snapshots after the last stored one and appends (never rebuilds), adding new entrants to the
   union while preserving the survivorship-free history. New union names are pulled by the next
   `update_daily_bars`.

4. **Data-availability timing.** BaoStock publishes a day's EOD bar after close; run the driver
   after ~15:30 CST on a trading day. A run before publication re-computes through the
   last available bar (idempotent, harmless).

5. **Execution model is preserved.** The signal is read at the month-end close and executed at the next
   trading day's close, T+1, 100-share lots, full A-share frictions — identical to the backtest.

## Capital tiers — the small-account constraint is material

`avg_names_held` (effective diversification) vs the 10-name target, deployed strategy,
2015-01 → 2026-06 (net of A-share frictions; `strategy.CAPITAL_TIERS`):

| band   | tier (¥)  | avg names | total return | maxDD  | note |
|--------|----------:|----------:|-------------:|-------:|------|
| small  | 10,000    | 9.6       | +116%        | -28.5% | marginal — ¥5 min commission + lot rounding drag CAGR ~3pp (≈7%/yr) |
| small  | 30,000    | 9.9       | +164%        | -30.7% | viable floor |
| small  | 50,000    | 9.9       | +173%        | -32.2% | viable |
| medium | 100,000   | 9.9       | +181%        | -32.6% | working regime |
| medium | 500,000   | 9.9       | +192%        | -33.0% | working regime |
| large  | 1,000,000 | 9.9       | +193%        | -33.0% | saturated (capacity reference) |
| large  | 5,000,000 | 9.9       | +194%        | -33.1% | saturated (capacity reference) |

Paper and live should start at ≥¥30k. At ¥10k the book is nearly full (9.6/10), but the ¥5
minimum commission and 100-share lot rounding drag the CAGR ~3pp (≈7% vs ≈10%); below ~¥5k it
cannot hold 10 names at all. The strategy saturates by ~¥100k (the large tiers add no new
behaviour and assume negligible, unmodeled market impact). Tier-by-tier paper trading exists
to establish this floor concretely before capital is committed.

## Operations

```
python scripts/paper_live.py                # refresh to today, step all tiers, report + persist
python scripts/paper_live.py --no-refresh    # recompute on the current lake (monitoring)
python scripts/paper_live.py --as-of 2026-03-31 --no-refresh   # historical replay to a date
```

Outputs (gitignored) under `results/paper/`: `curve_<tier>.parquet`, `trades_<tier>.parquet`,
`report_<tier>.json`, `logs/paper_<date>.log`. Every run is idempotent (recompute-from-seed),
so a missed or repeated day is harmless.

Scheduling — use the wrapper `scripts/paper_live.ps1` (captures stdout/stderr to a timestamped
log and propagates the exit code; Task Scheduler discards output otherwise), weekdays after close:

```
schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 15:35 /TN hermes-paper ^
  /TR "powershell -NoProfile -ExecutionPolicy Bypass -File F:\hermes-quant\scripts\paper_live.ps1"
```

### Unattended-operation guardrails (so the auto-record can be trusted)
- **Fail loud on a degraded pull**: `feed.update_daily_bars` raises (nonzero exit, no report
  written) if the BaoStock pull falls below 98% OK — a partial outage would otherwise leave a
  mixed-forward-adjusted-basis lake; the next clean run re-pulls the whole union and self-heals.
- **Fresh-vs-stale signal**: each report carries `run_date`, `lake_lag_days`, and `fresh`; a
  holiday/weekend/source-lag run (which idempotently re-computes the prior bar) prints a `STALE`
  banner rather than presenting itself as a fresh trading-day update.
- **Atomic writes**: all parquet/JSON outputs (lake bars, membership, curves, reports) write to a
  temp file then `os.replace()`, so a crash mid-write cannot wedge later runs with a truncated file.
- **No spurious rebalance at the right edge**: the engine's `+1 < n` guard and the current-month
  membership exclusion mean a non-trading-day run never fires a rebalance (verified).

## Deferred (not in this stage)

Corporate-action cash accounting (see above); a marks-only incremental pull on non-rebalance days
(the daily full re-pull is correct but heavier than needed); suspension-vs-delisting flag at the
right edge (latent, zero current impact); price-limit no-fill in the engine (justified for liquid HS300;
needed for a CSI 500 universe — see risk_control.md); the vnpy realtime gateway / `vnpy_paperaccount`
path (`execution/`), reserved for higher-frequency or true-live execution.
