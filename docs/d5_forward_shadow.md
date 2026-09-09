# The D=5 forward shadow — a historical champion facing a future it cannot be fitted to

Pre-registered as [issue #22](https://github.com/Rhymer-Lcy/hermes-quant/issues/22).
Module `src/hermes/live/shadow.py`; runner `scripts/paper_shadow_d5.py` and
`scripts/paper_shadow_d5.ps1`; tests `tests/test_shadow.py`.

**The existence of this shadow is not a recommendation of D=5.** Its only purpose is to expose a
historical maximum to data that cannot be selected after the fact.

## Why it exists, and why #21 did not support deployment

[Issue #21](rebalance_timing_study.md) enumerated all 31 calendar rebalance days and closed
**REJECTED / NOT IDENTIFIED**. D=5 was the historical CAGR champion — +12.67% against the canonical
D=1's +10.49% over 2015-2025, a gap of +2.18 pp/yr — and it passed seven of the eight pre-registered
gates. It failed exactly the gate that guards against enumerating 31 candidates:

- paired monthly difference +0.1497%/month, Newey-West HAC **t = 1.508**, raw p = 0.1316 —
  insignificant *before* any correction, Holm p = 1.0000 after;
- on the complete decade 2016-2025 the edge collapses to **+0.06 pp/yr**;
- regime rankings are unstable (mean Spearman rho +0.306; 2015-2018 vs 2019-2021 **negative**);
- a year-block bootstrap makes D=5 the winner in only **30.7%** of resamples, across 12 winners.

A historical maximum is not validated alpha. But it is also not disproved, and a backtest can never
settle it — only data that did not exist when the candidate was chosen can. That is what this
shadow collects.

## Why the shadow starts from a shared state on 2026-08-28

`SHADOW_INCEPTION_ASOF = 2026-08-28` — the latest lake bar when the experiment was frozen, with the
wall clock on the weekend of 2026-08-30, so the next trading bar had not yet occurred.

Both books replay the SAME history through the SAME engine from `PAPER_INCEPTION = 2026-06-18` on
the canonical schedule, and only executions **strictly after** the fork bar use D=5. At that bar the
two books therefore hold identical positions, identical share counts, identical cash and identical
equity in every tier, and both records are normalised to 0% return. Every later difference is
genuine forward evidence.

### Why the 2026-06-18 → 2026-08-28 counterfactual is NOT part of this record

A retrospective replay of that window under D=5 already exists in
`results/backtests/counterfactual/` (where D=5 trailed D=1 in all seven tiers). It is
**description, not a forward record**, and splicing it onto the front of the shadow curve would be
a methodological error, not merely untidy: it would hand D=5 a *differently chosen book* at
inception — a book selected with knowledge of how those two months turned out — and destroy the
like-for-like comparison. The shadow inherits the canonical book, not a counterfactual one.

## File layout

```
results/
  paper/                     # CANONICAL D=1 forward record -- never written by the shadow
    report_<tier>.json  curve_<tier>.parquet  trades_<tier>.parquet  logs/
  paper_shadow/
    d5/                      # the shadow, physically separate
      report_<tier>.json     # includes the D=1 comparator at the same as_of
      curve_<tier>.parquet   # equity FROM the fork bar onward
      trades_<tier>.parquet  # fills AFTER the fork bar only
      manifest.json          # frozen experiment definition
      logs/
```

Seven tiers throughout (10k / 30k / 50k / 100k / 500k / 1M / 5M), matching the canonical account —
the tier spread is what exposes 100-share lots, the CNY 5 minimum commission, achievable
diversification and cash drag, and a single representative account would hide all of it.

Each shadow report carries its own baseline comparator (`d1_equity_same_asof`,
`d1_return_since_shadow_inception`, `excess_return_vs_d1`, `drawdown_diff_vs_d1`,
`cost_diff_vs_d1`), so one JSON answers the whole question without cross-referencing.

## Blocking gates, checked on every run

1. **Production parity** — `live_step(persist=False)` reproduces the on-disk canonical reports in
   all seven tiers.
2. **Common-state parity at the fork bar** — the shadow path reproduces the canonical ledger
   exactly through 2026-08-28: equity, cash, positions, per-name shares, trade count and costs.
   Its reference is `live_step(as_of=2026-08-28)`, the canonical production book **at that bar**.
   *Corrected 2026-09-09.* It originally read the on-disk `results/paper/report_<tier>.json`,
   which the daily run advances: from the first trading day after inception the gate was comparing
   a book truncated at the fork bar against a report at the latest bar, so it failed every day and
   **the shadow recorded no forward evidence between 2026-08-31 and 2026-09-08**. The isolation
   held exactly as designed throughout — gate 1 passed, nothing was written, the exit code was
   discarded, and `results/paper/` stayed byte-for-byte identical — so nothing was corrupted; only
   eight days of shadow observations were lost, and they are recoverable because both books are
   recompute-from-seed. Gate 1 still compares the production path to the on-disk ledger at the
   latest bar, so the chain "shadow == production at the fork" and "production == ledger" is intact.
3. **No look-ahead** — every post-fork rebalance has `signal_bar < execution_bar`, reading only the
   PIT membership, PE, close and reversal history available at the signal bar.

Any failure exits nonzero **having written nothing**.

## Isolation, enforced rather than intended

- The runner fingerprints every file under `results/paper/` before it starts and re-checks
  afterwards; a change raises `ShadowGateError`. `tests/test_shadow.py` covers both the real
  persist path and a deliberately tampered fingerprint.
- The shadow never refreshes the lake. The canonical run owns the data pull; the shadow reads what
  it already validated.
- `paper_live.ps1` invokes the shadow **only** when the canonical run exits 0, and **discards its
  exit code**. A shadow failure cannot roll back, overwrite, delay or invalidate the canonical
  record — verified by running the hook against a missing script.
- `DEPLOYED` and `PAPER_INCEPTION` are unchanged.

## Provenance: four SHAs, deliberately separate

Schema v1 carried a single `created_at_commit = ca0d64b`. Forensics established what that actually
meant: the manifest was written at **12:48:57** on 2026-08-30 while HEAD was `ca0d64b` (committed
**12:18:01**), and every shadow source file was still an **uncommitted working-tree file** -- they
landed at **12:53:43** as `8f92a01` / `0eb165a` / `03acb02` / `b1a1fd0`. So the field recorded
"git HEAD at freeze time, on a dirty tree": the pre-implementation baseline, never the
implementation. One field was carrying three meanings.

Schema v2 separates them. The v1 value is preserved verbatim in `legacy_created_at_commit` and the
original file kept as `manifest_v1_original.json`; nothing was discarded.

| field | meaning |
|---|---|
| `preregistration_base_sha` | repository HEAD when issue #22 was registered and the manifest first frozen — `ca0d64b` |
| `initial_implementation_sha` | the first committed revision at which the three blocking gates can be executed end to end — **`03acb02`**, which added `scripts/paper_shadow_d5.py` hosting them (`0eb165a` added the module but not the runner) |
| `experiment_freeze_sha` | the clean, CI-green HEAD that is the audit baseline for all future forward evidence |
| `runtime_code_sha` | recorded per **report**, not in the manifest: the HEAD that computed that particular run |

`runtime_worktree_clean` accompanies the runtime SHA. **A persisted forward run refuses to proceed
from a dirty worktree** — evidence that cannot be traced to a revision is worse than no evidence.
A `--dry-run` may proceed, since it writes nothing. Either way the canonical D=1 record is
unaffected: the shadow's exit code is discarded by `paper_live.ps1`.

Provenance is code history, not strategy definition. A runtime SHA moves with maintenance; the
candidate day does not. The migration changed no experiment parameter — candidate D=5, baseline
D=1, inception 2026-08-28, `PAPER_INCEPTION`, strategy spec, cost model, tiers and schedule
semantics all carried over unchanged, and `validate_manifest` refuses any frozen-field drift.

## The manifest

`results/paper_shadow/d5/manifest.json` freezes the candidate day, the baseline day, the issue, the
four provenance SHAs above, both inceptions, the strategy spec, the cost model and the tier list, plus three explicit
declarations: the pre-inception counterfactual is not part of the forward record; the candidate
parameters are frozen; `results/paper/` is canonical and must not be rewritten. Every run
*validates* it and refuses to proceed on frozen-field drift. **A normal run never rewrites it** —
changing the experiment has to be a deliberate, visible act.

## Running it

```
python scripts/paper_shadow_d5.py                # gates, then persist (no data refresh)
python scripts/paper_shadow_d5.py --dry-run      # gates + report, write nothing
python scripts/paper_shadow_d5.py --as-of 2026-09-30
```

**A note on the next execution dates.** D=1's next rebalance is expected on 2026-09-01 and the
shadow's on 2026-09-07 (the 5th is a Saturday). These are **projections from the current trading
calendar, not established facts**: the lake ends 2026-08-28, so no September bar exists yet, and a
statutory closure could move either. The frozen rule is unchanged and is what governs — nominal
anchor = the 5th, execution = the first real trading day at or after it, rolling forward.

Daily automation: the existing `hermes-paper` scheduled task (weekdays 19:00 Asia/Shanghai) runs
`paper_live.ps1`, which after a successful canonical step calls `paper_shadow_d5.ps1`. No second
scheduled task is registered — one fewer thing to conflict, and one fewer thing to forget.

**To disable the shadow**: set `HERMES_SHADOW_D5=0` (the wrapper logs the skip and exits 0), or
delete the `Invoke-ShadowD5` call in `paper_live.ps1`. The canonical path is untouched either way.
**To re-enable**: unset the variable. **To retire the experiment**: delete
`results/paper_shadow/d5/` and note the outcome on issue #22 — do not quietly repurpose it for a
different candidate day.

Both records are idempotent (recompute-from-seed): rerunning a day reproduces it byte-for-byte
apart from the wall-clock fields, so a missed or repeated run is harmless.

## Evaluation schedule, frozen in advance

| horizon | what is permitted |
|---|---|
| daily | descriptive reporting only — returns, difference, drawdown, costs, holdings differences |
| 12 months | one interim review, descriptive; **may not replace D=1** |
| **24 complete comparable monthly cycles** | the primary forward verdict |

Forbidden: stopping early on a win; changing the day on a loss; broadening to D=3-9; redefining the
start; dropping unfavourable months. **There is no "keep looking for a better day" here** — D=5 was
fixed before any forward result existed, and adding candidates later would recreate exactly the
31-choose-1 problem #21 was built to avoid.

At 24 months the report will cover cumulative net return, annualised return, paired monthly net
return difference, Sharpe, Calmar, max drawdown, turnover, total costs, share of months won, and
consistency across all seven tiers — as a **single pre-committed comparison**, so no new
multiplicity arises. The sample will still be thin: 24 monthly observations support very little,
which is recorded here in advance so it cannot be forgotten when the numbers arrive.

## The line that governs everything else

**Shadow performance can never rewrite canonical D=1 history.** Whatever D=5 does, the forward
record of what the deployed strategy actually did since 2026-06-18 is the ledger under
`results/paper/`, and it stands.
