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

## Verdict

GO. Phase 1 (not started): build the lake -- universe join, daily bars, premium series, the
two cross-checks -- then run the pre-registered study exactly as frozen above.
