# Calendar-anchor rebalance study — C. ISOLATED HISTORICAL WINNER, keep D=1

Pre-registered as [issue #23](https://github.com/Rhymer-Lcy/hermes-quant/issues/23).
Script `scripts/rebalance_anchor_day_study.py`; schedule helper
`src/hermes/research/backtest/schedule.py`; tests `tests/test_anchor_schedule.py`; results under
`results/studies/rebalance_anchor_day/`; figure `results/figures/rebalance_anchor_day.png`.

## This supplements [#21](rebalance_timing_study.md); it does not repeat it

#21 already swept D=1..31 and closed REJECTED / NOT IDENTIFIED. A specification review found four
genuine gaps, one of which was a commitment #21 made and did not keep:

| gap | #21 | here |
|---|---|---|
| **semantics** | execution CLAMPED to the anchor's own calendar month | cross-month **roll-forward allowed**; identical for D=1..23, differs for D=24..31 (D=31 on 81 bars) |
| **window** | 2015-2025 and 2016-2025 | a true trailing decade **2016-08-29 .. 2026-08-28**, including eight months that did not exist when #21 ran |
| **rolling stability** | **pre-registered and never implemented** (`rolling` appears 0 times in its script, and the omission was not disclosed) | delivered |
| **attribution** | never asked why days differ | holdings overlap, names changed, extreme-month concentration |

## Confirmed legacy semantics

`portfolio.py` computes `period_end` as each calendar month's last trading bar and executes at
`pos+1`, i.e. signal at the month-end close, execution at the next trading day's close. In execution
terms that is anchor day 1 — verified, not assumed.

## Anchor definition (frozen before any result)

1. `anchor_date(M,d) = date(y, m, min(d, days_in_month))` — D=31 resolves to Apr 30, Feb 28, Feb 29.
2. execution = the first **actual trading bar >= anchor_date** from the project's real calendar,
   **cross-month roll-forward allowed**. No weekday arithmetic.
3. signal = the immediately preceding trading bar, strictly earlier.
4. at most one rebalance per anchor month; execution priced at that bar's close.

The full per-month mapping is persisted to `schedule.csv`. Cross-month executions occur only for the
late anchors: D=24/25/26 one each, D=27 three, D=28 six, D=29 sixteen, D=30 thirty-one, D=31 forty.

## Day-1 parity gate — passed exactly

| | |
|---|---|
| schedules identical to `month_end_schedule` | True |
| rebalances / trades | 120 / 1,015 |
| equity max abs difference | **0.000e+00** |
| costs | 10,764.37 vs 10,764.37 |

## Window, capital, frictions

2016-08-29 .. 2026-08-28 (2,428 bars, 121 months), identical for all 31. Signals computed on the
**full** history then sliced, so no candidate pays a warm-up penalty. Primary ranking basis
**CNY 100,000 net of frictions**; robustness at 30k and 500k; a zero-cost run is diagnostic only.
Everything except the anchor is the deployed spec, unchanged.

## Results — the headline, and why it is not the answer

**Baseline D=1: CAGR +7.93%, Calmar 0.258, maxDD −30.80%, rank 8/31 by CAGR.**

| D | CAGR | ΔCAGR | Calmar | maxDD | costs |
|---|---:|---:|---:|---:|---:|
| **5** | **+9.13%** | **+1.20%** | **0.292** | −31.30% | 12,428 |
| 6 | +8.87% | +0.94% | 0.271 | −32.78% | 12,367 |
| 4 | +8.69% | +0.76% | 0.268 | −32.42% | 12,001 |
| 3 | +8.60% | +0.67% | 0.271 | −31.74% | 11,641 |
| 1 (baseline) | +7.93% | — | 0.258 | −30.80% | 10,764 |
| 26 (worst) | +4.09% | −3.84% | 0.104 | −39.23% | 9,778 |

Spread best-to-worst **5.05 pp**. All 31 rows, gross and net, are in
`results/studies/rebalance_anchor_day/summary_by_day*.csv`.

Three things point *toward* the effect being real, and they are stated first because they are the
strongest case that can be made for it:

- **It is a plateau, not a spike.** D=3,4,5,6 all beat D=1; D=2 and D=7 do not. A four-day
  contiguous early-month region.
- **It is not a cost artefact.** The winners trade *more* (D=5 pays 12,428 against D=1's 10,764) and
  still win; the gross sweep has the same shape.
- **It survives the split and the tiers.** Cross-half rank correlation **rho +0.710** (against #21's
  +0.306); D=5 is rank 1 in the discovery half and **rank 3/31 in the validation half**; D=5 leads at
  30k, 100k and 500k alike.

## Why the verdict is nevertheless C

**Removing two calendar years destroys the result, and reverses its sign.**

| D=5 compounded edge vs D=1 | |
|---|---:|
| all eleven years | **+25.08%** |
| excluding 2018 | +8.61% |
| excluding 2023 | +8.89% |
| **excluding 2018 and 2023** | **−7.95%** |
| those two years alone | +14.06% |

Nine of eleven years leave D=5 **behind** D=1. The annual table shows why: D=5 beats D=1 by +6.9 pp
in 2018 and +7.2 pp in 2023, while 2019 (−3.9), 2024 (−5.2) and 2026 (−4.1) run the other way.

**And the rolling-window figure cannot rescue it, because those windows are not independent.**
D=5 beats D=1 in 87% of 15 rolling 3-year windows — but the windows are stepped 6 months across a
10-year span, so **13 of the 15 contain 2018 or 2023**, and at most **three disjoint** 3-year windows
exist in the period. The high share is the same two years propagating through overlapping windows,
not fifteen independent confirmations. This caveat is printed by the script itself.

### A defect in this study's own verdict logic, found and fixed

The pre-registration required (§9.4) an answer to "is the apparent advantage generated by one or two
extraordinary years?" The drop-one-year table was computed and printed — and then **not consulted by
the verdict branch**, which returned "B. BROAD ROBUST WINDOW" on five passing checks while the
year test sat unread beside it. The branch now takes year-robustness as a necessary condition for
B and D, and the classification changed to C. This was honouring a pre-registered requirement that
had been computed and ignored, not adding a hurdle after seeing the result.

## Economic attribution (gap 4)

- **Execution dates**: D=1 and D=5 each rebalance 120 times; only **21 dates coincide**, and every
  one falls in a month with a statutory long closure that collapses the 1st and the 5th onto one bar.
- **Holdings overlap** across all 2,425 bars: median Jaccard **0.82**, mean 0.85, minimum 0.00;
  median **2 names** differ per bar, maximum 10. The two books are close but not the same.
- **Turnover / costs**: D=5 trades 9.16 m against D=1's 7.49 m and pays 15% more in costs.
- **Concentration**: the six largest-|delta| months sum to **+0.55%, only +2% of the +25.08%
  terminal gap**; the other 114 months carry +8.66%. So the edge is *not* a handful of lucky trades —
  it is many small monthly differences that happen to **cluster inside two calendar years**. Those
  are different failure modes, and only the second one applies here.

## Verdict: C. ISOLATED HISTORICAL WINNER

D=5 is **the highest-CAGR candidate on this window**, not an optimal day. It shows a plateau, passes
the split, the tiers and the gross check, and still fails the test that matters most: its entire
advantage lives in 2018 and 2023, and without them it trails the baseline.

Searching 31 alternatives on one decade guarantees a maximum. That D=5 tops both #21's window under
clamping and this window under roll-forward is worth noting — but the two windows overlap by nine
years, so this is close to the same evidence twice, not independent confirmation.

## Recommendation

**Keep D=1.** It is the simplest and most auditable rule, it sits inside the flat early-month region
rather than at an extreme, and nothing here beats it on evidence rather than hindsight.

No new shadow is opened. The D=5 forward shadow from [#22](d5_forward_shadow.md) already exists,
frozen at the 2026-08-28 common state, and **it is exactly the right instrument for this question**:
it will accumulate months that no backtest can be fitted to. This study adds one prior to it —
that D=5's historical record depends on two years — which will make the forward evidence easier to
read either way.

If the early-month region is ever revisited, the object of interest is the **region D=3-6**, never a
single date, and only after the shadow has run its pre-registered 24 cycles.

## What was NOT changed

`DEPLOYED`, `PAPER_INCEPTION`, the canonical `results/paper/` ledger and the scheduled production
tasks are untouched. No history was recomputed to pretend a different anchor was ever deployed.
Market-state adaptive timing remains out of scope.

## Reproduce

```
python scripts/rebalance_anchor_day_study.py     # ~4 min; parity gate asserts before any sweep
pytest tests/test_anchor_schedule.py
```
