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
dominant silent source of alpha decay). `scripts/paper_dryrun.py` asserts that the ledger equity equals the
engine equity bar-for-bar at every tier; if that gate fails, paper has drifted from research.

Going forward is one daily step (`live.paper.live_step`): refresh the lake to today, recompute
scores with the same factor code, and replay, extending the ledger by the new day(s).

### One source of truth for the strategy

The deployed spec lives in exactly one place, `live/strategy.py` (`DEPLOYED` +
`deployed_signal`), imported by both the research demo and the live driver. Neither
re-spells the 5:1 value/reversal blend, so they cannot diverge. `test_paper.py` locks
`deployed_signal` to the documented blend.

## Forward-only rigor (things the backtest never had to face)

1. **Forward-adjusted re-basing is not blindly append-only.** Forward-adjusted prices rescale the
   entire history whenever a dividend/split occurs, so naively appending new days would mix two
   adjustment bases in one series. `live.feed.update_daily_bars` therefore refreshes INCREMENTALLY
   (`ingest.pull_universe_incremental`): it appends each code's new bars but full-re-pulls any code
   whose last 60 stored closes (`ingest.REBASE_WINDOW_BARS`) changed on re-pull (the signal of a
   re-base), keeping the lake on one basis at a fraction of the cost; `replay` then recomputes the
   ledger wholesale from the seed, so the equity curve is self-consistent and re-running a date on
   the same lake reproduces it. Until 2026-10-25 only the last stored bar was compared, which missed
   every dividend factor the vendor applied after the ex-date bar had been appended; the record was
   restated on that date (see "Restatement of 2026-10-25" below). Forward-adjusted
   prices reinvest dividends via the adjustment, so the paper curve approximates a total-return account.
   **That reinvestment is idealised, and the size of the idealisation is now measured rather than
   asserted** (issue #24, [docs/dividend_tax_study.md](dividend_tax_study.md)): the adjusted series
   reinvests every dividend at the ex-date close, instantly, PRE-TAX, in fractional shares. Applying
   the statutory A-share differentiated dividend tax (Caishui [2015] No. 101) to the book's own FIFO
   holding periods costs **+0.30 pp/yr** — about 3% of the strategy's net return, one-signed. So the
   paper curve is a pre-dividend-tax figure for an individual investor, and a corporate holder is
   exempt. The two remaining pieces — dividend cash timing and reinvestment lot friction — were
   measured in the same study and are genuinely second-order (**±0.008 pp/yr**); that adjective was
   previously applied to the tax as well, and for the tax it was wrong.

2. **Suspension vs delisting at the right edge.** The engine force-liquidates a holding once its
   price series ends (NaN after `last_valid_index`), at the last real price. A suspension does not
   trigger this: BaoStock serves a suspended day as a row carrying the previous close (37,802 rows
   with zero or no volume across the 667 HS300-union files on 2026-10-26, none with an empty close),
   so a suspended holding stays held and valued. What does trigger it at the right edge is an
   ABSENT bar: a delisting (correctly), but also a pull that failed for that name or a bar not yet
   published, where the exit would be a sell the market never printed. On the live path every such
   exit (`right_edge_exits` in the report) must be a delisting BaoStock itself records
   (`stock_basic` status 0, `outDate` on or before the session), or nothing is written and the run
   exits 75 (`feed.assert_right_edge_safe`; see Operations). Past bars and the research backtests
   keep the engine's rule unchanged.

3. **Membership must stay current.** `live.feed.extend_membership` pulls HS300 month-end
   snapshots after the last stored one and appends (never rebuilds), adding new entrants to the
   union while preserving the survivorship-free history. New union names are pulled by the next
   `update_daily_bars`.

4. **Data-availability timing.** BaoStock posts a day's EOD bars over ~2-3 h after close, so the
   task runs in the evening. Which session the record must hold is decided in Beijing time from a
   validated exchange calendar (`live.session`): today once its publication is due (18:00), else the
   previous session. If the lake does not reach that session, or fewer than 90% of current members
   carry its bar, `refresh` raises and no report is written (retry once published). The former
   check measured coverage on the lake's own latest date, so a lake that had received nothing new
   passed it.

5. **Execution model is preserved.** The signal is read at the month-end close and executed at the next
   trading day's close, T+1, 100-share lots, full A-share frictions — identical to the backtest.

## Capital tiers — the small-account constraint is material

The table below is the full-history BACKTEST characterization of each tier (`avg_names_held` =
effective diversification vs the 10-name target, plus total return / maxDD over 2015-01 → 2026-06,
net of A-share frictions; `strategy.CAPITAL_TIERS`). It is the rationale for the tier choice, NOT
the live paper record (see "Live paper record" below). These `avg_names_held` reflect the GROWN
capital over the backtest; a FRESH seed is tighter — a new ¥10k holds only ~7 of the 10 names at
current prices, so its starting friction is worse than the 9.6 shown here:

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

**Two distinct records.** The same deployed strategy yields two curves — keep them separate:

| record | command | seed | measures | written to |
|---|---|---|---|---|
| **paper (forward)** | `paper_live.py` | full capital at `PAPER_INCEPTION`, then forward | return SINCE inception (facing the unknown) | `results/paper/` |
| **backtest (research)** | `paper_live.py --backtest` | full capital at 2015-01 | the 2015→ historical record | `results/backtests/` |

Both compute identically (recompute-from-seed, one engine); they differ only in the seed date. The
signal always uses full history so factor lookbacks are satisfied — only the *ledger* is seeded at
inception. The scheduled daily run is the forward paper record.

## Operations

```
python scripts/paper_live.py                # refresh, verify the session, step all tiers, persist
python scripts/paper_live.py --no-refresh    # recompute on the current lake; persists only if it holds the session
python scripts/paper_live.py --as-of 2026-03-31 --no-refresh                  # replay, print only
python scripts/paper_live.py --as-of 2026-03-31 --no-refresh --out-dir <dir>  # replay into <dir>
```

A `--as-of` replay never writes `results/paper/` (it used to overwrite the canonical files with a
ledger cut at that date); `--out-dir` may not point there. It also requires `--no-refresh`, so a replay
never pulls BaoStock, extends the membership or rewrites any lake file.

Outputs (gitignored) under `results/paper/`: `curve_<tier>.parquet`, `trades_<tier>.parquet`,
`report_<tier>.json`, `logs/paper_<date>.log`. Every run is idempotent (recompute-from-seed),
so a missed or repeated day is harmless.

Scheduling — use the wrapper `scripts/paper_live.ps1` (captures stdout/stderr to a timestamped
log and propagates the exit code; Task Scheduler discards output otherwise), weekdays in the evening
(~19:00, after BaoStock posts the day's EOD bars):

```
schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 19:00 /TN hermes-paper ^
  /TR "powershell -NoProfile -ExecutionPolicy Bypass -File <repo>\scripts\paper_live.ps1"
```

### Unattended-operation guardrails (so the auto-record can be trusted)
- **VPN-resistant connection (resolve + pin the server IP)**: the BaoStock client connects to the
  hostname `public-api.baostock.com:10030`. A split-tunnel corporate VPN (observed: Sangfor SSL VPN)
  can break the default DNS resolver for that host — `getaddrinfo` returns `gaierror` — while leaving
  the server's physical route intact and reachable. Login then fails with a `10002xxx` network error.
  The fix needs no admin rights and no VPN reconfiguration: the wrapper resolves the host via a PUBLIC
  DNS server (`Resolve-DnsName -Server 223.5.5.5`, then `119.29.29.29`, `8.8.8.8` — the VPN does not
  block UDP/53 to these) and exports the IP as `HERMES_BAOSTOCK_IP`; `baostock_source.session()` pins
  the client's socket to that IP, skipping DNS entirely. Standalone (`python scripts/paper_live.py`)
  the same `session()` resolves via the OS (caching the result under `data/cache/`) and falls back to
  the last cached IP, then a seed. The scheduled run therefore succeeds with the VPN on.
- **Retry transient failures with backoff**: a residual TRANSIENT data failure — a server-side blip
  during the heavy publication window, a momentary network drop, or a mid-publication run — exits 75
  (`EX_TEMPFAIL`, raised as `BaoStockUnavailable`); the wrapper then retries on a fixed interval, up
  to `HERMES_RETRY_MAX` attempts spaced `HERMES_RETRY_DELAY_SEC` apart (default 24 × 300 s ≈ 2 h),
  self-healing without a manual re-trigger. A FATAL error (any other nonzero exit) returns
  immediately. Each run is recompute-from-seed, so retrying is safe.
- **Kill a silent attempt**: the BaoStock client reads its socket with no timeout, so a stalled
  server connection blocks an attempt forever (2026-10-22: 6.8 h until shutdown; 2026-10-24: a stack
  dump showed the wait in `baostock/util/socketutil.py`). The wrapper runs Python unbuffered and
  kills an attempt that writes nothing for `HERMES_ATTEMPT_IDLE_SEC` (default 1800 s), treating it as
  exit 75. The trigger is silence, not duration: the pull prints every 50 symbols, so a slow run that
  is still progressing is never killed. `HERMES_RETRY_BUDGET_SEC` (default 22 h) bounds the whole
  task, so a run can never overlap the next day's (the task is registered with
  `MultipleInstances=IgnoreNew`).
- **Fail loud on a degraded pull**: `feed.update_daily_bars` aborts (no report written) if the
  BaoStock pull falls below 98% OK — a partial outage would otherwise leave a
  mixed-forward-adjusted-basis lake. It raises `BaoStockUnavailable` (exit 75), so the wrapper
  retries; once the source recovers, the next pull re-pulls the whole union and self-heals.
- **Session evidence in every report** (a schema extension; existing keys unchanged):
  `evidence` is `current_session` (the lake holds the session due in Beijing time, verified against
  a validated calendar), `historical_recompute` (a `--as-of` replay or `--backtest`, never written to
  the canonical record), or, in the refused cases, `stale_source` / `incomplete_publication`, which
  are raised and never written. `expected_session`, `session_date`, `session_coverage` and
  `session_checked_at` (ISO 8601, +08:00) record what was checked; `right_edge_exits` lists names
  the engine sold today without a bar today. `run_date` / `lake_lag_days` / `fresh` are kept for
  compatibility and now count on the Beijing date; `fresh` (lag <= 4 calendar days) cannot tell a
  stale replay from a current record and is superseded by `evidence`. Reports written before this
  change lack the new keys.
- **Calendar, not a reference stock**: the trade calendar comes from BaoStock `query_trade_dates`,
  is validated (one row per day, no weekend session, no closure longer than 12 days) and is
  cross-checked against the lake's own bar dates; a contradiction is fatal (exit 1). The
  incremental pull no longer takes "the latest date" from sh.600000, whose staleness or suspension
  used to freeze the whole universe silently; every code's 60-bar window is checked on every run,
  which also catches a late adjustment factor on a same-day re-run.
- **No sell the market did not print**: see section 2. A held name without a bar on the session is
  written only as a BaoStock-recorded delisting; a failed pull or an unpublished bar blocks the
  write (exit 75), even when the pull as a whole is above the 98% threshold. All tiers are computed
  before any is written, so a refusal never leaves some tiers updated and others not.
- **No lake write may drop a stored bar**: a full-series write (`ingest.write_full_series`, used by
  `pull_universe` and the refresh's re-base re-pull) is refused if the new series lacks any date the
  stored file holds, which blocks an old `end` (the research builders default to 2025-12-31), a
  later `start`, or a vendor gap from truncating a current series. `scripts/ingest_union.py` has no
  default end date any more (`--end` is required) and exits 1 if any code was not written.
- **Atomic writes**: all parquet/JSON outputs (lake bars, membership, curves, reports) write to a
  temp file then `os.replace()`, so a crash mid-write cannot wedge later runs with a truncated file.
- **No spurious rebalance at the right edge**: the engine's `+1 < n` guard and the current-month
  membership exclusion mean a non-trading-day run never fires a rebalance (verified).
- **Reject incomplete publication**: BaoStock posts EOD bars over ~2-3 h, so the live refresh
  aborts (no report written) if fewer than 90% of current members carry the due session's bar —
  preventing a mid-publication run from mis-liquidating the not-yet-posted names. It raises
  `BaoStockUnavailable` (exit 75), so a run that lands mid-publication retries until publication
  completes. (`feed.assert_publication_complete`, the older check on the lake's own latest date, is
  kept only for an explicit historical `refresh(end=...)`.)

## Restatement of 2026-10-25 (lake adjustment basis)

**What was wrong.** A read-only audit on 2026-10-24 re-pulled every HS300-union series from
2026-01-01 and compared it with the lake bar by bar: 346 of the 643 series with 2026 bars were on a
mixed adjustment basis. 281 had missed exactly one 2026 dividend factor and 65 more than one; the
last off-basis day of the single-factor codes fell in 2026-06 (47), 07 (130), 08 (34), 09 (51) and
10 (19). The missing factor had a median of 1.36% and a maximum of 45.0%. In each such series the
lake recorded the dividend as a price drop, so the record booked it as a loss. 8 of the 10 names
held on 2026-10-23 were affected. Cause: the one-bar re-base check described in section 1.

**What was done.** The check now compares a 60-bar window (`f488f0d`); on the 2026-10-24 lake it
flagged exactly the 194 codes the audit predicted for that window and nothing else. Because older
misses lie outside any window, the whole union was re-pulled in full on 2026-10-25. The re-pulled
lake reproduced the audit code for code (the same 346 series rescaled, the other series unchanged)
and changed no date and no non-price column. `scripts/audit_lake_basis.py` repeats the audit.

**Effect on the record (as_of 2026-10-23, total return since inception and maximum drawdown):**

| tier | return before | return after | max DD before | max DD after |
|---:|---:|---:|---:|---:|
| 10,000 | -2.28% | +0.59% | -3.82% | -3.91% |
| 30,000 | -0.37% | +2.96% | -5.04% | -4.90% |
| 50,000 | -1.98% | +1.41% | -6.53% | -5.72% |
| 100,000 | -1.96% | +1.54% | -8.19% | -7.20% |
| 500,000 | -2.04% | +1.76% | -9.18% | -8.08% |
| 1,000,000 | -1.91% | +1.79% | -9.16% | -8.11% |
| 5,000,000 | -1.91% | +1.85% | -9.19% | -8.15% |

The deployed top-10 is identical on both lakes on every signal date (2026-06-18, 06-30, 07-31,
08-31, 09-30), and the names held after each of the 33 tier-rebalance points are the same; what
changed is the dividend accounting and the integer-lot share counts, which the engine sizes on
forward-adjusted prices, so share counts differ from the inception fill onward. The
pre-restatement ledgers and the mixed-basis lake are archived outside the repository.

**What this restatement is and is not.** No parameter, date or rule changed; the same code and the
same seed produce the restated record from corrected inputs. The record was always defined as the
engine's output on the lake, and is so again. What it cannot recover is the daily attestation: the
bars through 2026-10-23 were written on 2026-10-25, and those of 2026-10-22 and 10-23 were already
written late, on 2026-10-24, after the scheduled runs of those evenings hung or exited without a
report (see "Kill a silent attempt" under Operations). Research studies that read 2026 bars (the #24
dividend-tax event sample runs to 2026-07-16; #25 uses the paper period) ran on the partly mixed
lake; their frozen verdicts were not re-run and the effect on them is not quantified.

## Canonical record vs forward shadows

`results/paper/` is the **canonical** forward record and the only one. It is what the deployed
strategy actually did, and nothing may rewrite it.

A *forward shadow* is a separate, frozen experiment racing the canonical record from a shared
state: same engine, same signal, same costs, same tiers, differing in exactly one pre-registered
parameter. The first is the D=5 rebalance-day shadow ([issue #22](d5_forward_shadow.md)), which
forked from the canonical book at the 2026-08-28 close after
[issue #21](rebalance_timing_study.md) found D=5 to be the historical champion but statistically
uncertified.

The hierarchy is strict, and enforced rather than merely documented:

- shadows write only under `results/paper_shadow/<id>/`, never into `results/paper/`;
- the shadow runner fingerprints the canonical tree before and after each run and raises if it moved;
- `paper_live.ps1` invokes a shadow **only** after the canonical step exits 0, and **discards** its
  exit code, so a shadow fault cannot delay, roll back or invalidate the canonical record;
- a shadow never refreshes the lake -- the canonical run owns the data pull;
- disabling a shadow (`HERMES_SHADOW_D5=0`) leaves the canonical path byte-identical.

**A shadow's performance is evidence about a candidate, never a restatement of history.**

## Deferred (not in this stage)

Corporate-action cash accounting (see above); a membership-aware suspension-vs-delisting
rule at the right edge for genuine multi-day suspensions (the acute same-day-publication case is now
guarded — see Operations); price-limit no-fill in the paper ledger (the backtest engine
offers it as an opt-in, `limit_block`, used by the CSI 500 studies — see risk_control.md; the
canonical ledger and the D5 shadow run without it, and how often it would bind on HS300 has
not been measured); the vnpy realtime gateway / `vnpy_paperaccount`
path (`execution/`), reserved for higher-frequency or true-live execution.
