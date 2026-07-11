# CB lake phase 0: the free-data gate and the pre-registered double-low question

Convertible bonds (CB) are the one A-share segment where a retail account trades T+0 with no
stamp tax, and where the canonical retail rule -- "double-low" (buy the bonds cheapest in price
plus conversion premium) -- has a published live-money following. That popularity is exactly why
the claim needs an adversarial test: the rule's easy years predate the 2022-08 exchange trading
rules (both exchanges imposed +-20% limits and monitoring) and the 2023 credit events (Soute and
Landun defaulted and were delisted -- the first principal losses in the modern CB market). A
backtest that omits the dead bonds or reconstructs conversion prices from today's values would
manufacture the answer.

Phase 0 is a kill-first gate, in the same spirit as the survivorship and look-ahead audits
elsewhere in this repo: before any lake is built, verify that FREE sources can reconstruct the
three point-in-time ingredients, and freeze the research question so later results cannot shape
it. Reproduce: `python scripts/probes/probe_cb_freedata.py`.

## Pre-registered question (frozen 2026-07-11, before any backtest exists)

Did double-low survive the 2022-08 trading-rule change and the 2023 credit events, net of
costs, on a survivorship-free universe? Frozen design, no tuning after results:

- Window 2018-01 -> present (earlier the universe is too thin: 189 bonds by end-2018, 620 by
  end-2021). Monthly rebalance at close, equal weight, top-20 by double-low score = CB close +
  conversion premium x 100; top-10 reported once as the small-capital variant.
- Universe at each rebalance: bonds alive with >=60 prior trading days and above a 20-day median
  turnover (close x volume) floor; the floor is calibrated to cut the thinnest decile and is
  fixed BEFORE any return series is computed. No credit-rating filter (no PIT rating history).
- Costs 0.10% round trip (commission + fees, no stamp tax), sensitivity at 0.20%. Suspended on
  rebalance day = held, traded at the next tradable close.
- A bond that stops trading exits at its last main-board close. For the two defaults that is
  OPTIMISTIC (the delisted-segment tail is not in free data), so defaults are also reported
  marked to zero as the stress variant.
- Survival verdict requires ALL of: net return above the equal-weight CB universe over the full
  window; net positive over the 2022-08 -> 2024-12 stress sub-window; drawdown not more than
  5 pp worse than the universe.

## Gate results (probe run 2026-07-11 -- all pass)

| gate | source | finding |
|---|---|---|
| A universe | Eastmoney listing | 1024 listed bonds back to 2007, incl. both 2023 defaults (128100, 123015), forced exits and forced calls; carries underlying-stock code, issue size, listing date |
| B dead-bond bars | Sina daily | full life served for all four probed dead bonds (e.g. Landun 1132 bars 2018-09 -> 2023-07), OHLC + volume |
| C2 premium trail | Eastmoney value-analysis | dated close / conversion-value / premium series over the WHOLE life of dead and live bonds; live bond 1 day stale |
| C1 revision log | JSL (no login) | explicit downward-revision records (Landun all 5), with meeting date and effective date |

## Limitations recorded before any result exists

- Point-in-time REMAINING size is not free; the turnover floor stands in for it.
- The post-delisting 404xxx segment is not served (probe fact), hence the two-mark default
  treatment above.
- All three sources are scrapers, not authoritative feeds. The lake build must cross-check
  C2 close vs Sina close, and C2's implied conversion price vs C1's revision log, and record
  the mismatch rate before the study runs.

## Phase-0 verdict

GO -- and phase 1 was built and run the same day (below): the lake (`src/hermes/cb/data.py`,
`scripts/build_cb_lake.py`), the accounting engine (`src/hermes/cb/backtest.py`), the two
cross-checks (`src/hermes/cb/checks.py`), then the study exactly as frozen
(`scripts/cb_double_low_study.py`). Implementation choices the freeze left open were fixed
before any return was computed, in the study script's header: the rolling turnover median
needs >=10 traded days in its 20-day window, the floor percentile is taken per exchange,
score ties break by code order, and execution is at the day-after-signal close (the
index_effect_study convention -- the freeze said "rebalance at close" without pinning the
observation/execution lag, so the retail-replicable reading applies).

## Phase 1: the lake as built (2026-07-11)

1020 listed bonds (40xxxx delisted-segment re-codes dropped); bars and premium series both
served for 1005; the 15 misses are bonds the sources do not serve, ALL outside the study
window -- in-window coverage is 943/943 (one "miss", 127114, simply lists on 2026-07-14 and
has not traded yet). Cross-checks, run before any strategy return existed:

- close agreement (Sina bars vs Eastmoney value-analysis): 750,121 joined rows,
  0.0085% disagree beyond 0.5%, worst 2.97% -- two independent upstreams, same tape;
- revision agreement (EM conversion-value jumps vs the JSL log): 122-bond sample, 43 with
  a served log, 68 logged revisions of which 62 resolvable against the series -- 100%
  matched. Sampled bonds with no served log are excluded from the denominator ("never
  revised" and "not served" are indistinguishable), as pre-committed;
- turnover floors (p10 of the 20-day median of close x volume): SH 5.1e6, SZ 4.9e6 -- the
  two markets agree in magnitude, so the per-exchange split turned out precautionary;
- eligible bonds per rebalance: min 18 (early 2018), median 328, max 517, 103 rebalances.

## Phase 1: the pre-registered result (frozen design, run once, 2018-01 -> 2026-07)

| run | net total | CAGR | maxDD | stress net 2022-08->2024-12 | one-way turnover |
|---|---:|---:|---:|---:|---:|
| top-20 | **+167.3%** | +12.4% | −26.1% | **+16.7%** | 42.7%/mo |
| top-20, 2x costs | +155.9% | +11.8% | −26.3% | +15.5% | 42.7%/mo |
| top-20, defaults marked to 0 | +164.0% | +12.2% | −26.1% | +15.2% | 42.7%/mo |
| top-10 (small capital) | +158.3% | +11.9% | −33.4% | +20.8% | 50.6%/mo |
| equal-weight universe | +116.6% | +9.6% | −23.4% | −8.4% | -- |

All three pre-registered criteria PASS: net above the equal-weight universe; stress
sub-window net positive (+16.7% against the universe's −8.4%); max drawdown within 5 pp of
the universe (−26.1% vs −23.4%). **Verdict: double-low SURVIVED the 2022-08 trading rules
and the 2023 credit events, net, survivorship-free.** Robustness margins are wide: doubling
costs takes 11 pp off 167 pp, and marking both defaults to zero takes 3 pp.

What the survival is made of -- and what it does not establish:

- the edge is EPISODIC: 2019 (+60% vs +32%) and 2024 (+26% vs +2%), the two post-crash
  low-price recovery years, supply most of it; the strategy LAGS the universe in 2021,
  2023 and 2025. Whoever holds this must sit through multi-year stretches of losing to
  the benchmark;
- the drawdown trough is 2021-02-08 -- the post-Yongmei low-price CB crash -- and the
  universe's is 2024-08-22 (that year's credit scare): the curve carries both documented
  panics rather than sidestepping them;
- biases carried from phase 0: coupon cash is never credited (understates returns a
  little), forced calls exit at the last close (the notice-period crash is included),
  post-2022 limit-locked closes are still treated as fillable (not modeled);
- survival of a pre-registered backtest is NOT deployable alpha. The whole window is
  still history and the edge is regime-concentrated. Deployment would require its own
  forward paper record -- started below.

## Forward paper record (inception 2026-07-10)

The stage the equity strategy already runs (docs/paper_trading.md), applied to this line:
an EOD scheduled task (`hermes-cb-paper`, weekdays 19:40 Beijing, Beijing-anchored
trigger) refreshes the lake incrementally (bonds with a bar in the last 45 days plus new
listings), then recomputes the record from inception with the SAME signal code
(`cb/signals.py`) and the SAME engine (`cb/backtest.py`) the study ran -- no re-implementation,
no train/serve drift. Reports land under results/paper/ (cb_curve.parquet, cb_report.json);
driver: `python scripts/cb_paper_live.py`.

Frozen at inception, before any forward bar existed (`cb/paper.py`):

- inception 2026-07-10, the last close before the first unknown forward day; the seed
  enters the then-current top-20 at the NEXT trading day's close;
- signals at every later month-end close, execution at the next trading day's close; an
  in-progress month's provisional "month-end" has no next bar yet, so the engine waits
  until the true month-end is confirmed by construction;
- top-20 equal weight, 0.05% per side; turnover floors frozen at the study's calibration
  (SH 5,109,363 / SZ 4,867,041) -- forward eligibility must not recalibrate itself;
- a bond that stops trading exits at its last close (the study's primary mark).

Reading the record honestly: the study's edge is EPISODIC (concentrated in post-crash
recovery years), so multi-month stretches of losing to the equal-weight universe are the
expected path, not a failure signal; judge at 12 months minimum. Replicating the book
needs roughly 30k CNY (20 names x one 10-bond lot at typical prices, with slack). The
data chain is scrapers (Eastmoney/Sina): a failed evening self-heals by retry and by the
next run's recompute-from-inception.
