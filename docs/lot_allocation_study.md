# Integer-lot allocation: how faithfully can the intended book be executed?

Pre-registered as [issue #25](https://github.com/Rhymer-Lcy/hermes-quant/issues/25) before any
comparative number existed. Module `src/hermes/research/backtest/lots.py`; battery
`scripts/lot_allocation_study.py`; baseline diagnostic `scripts/lot_residual_diagnostic.py`;
tests `tests/test_lots.py`.

**Verdict: EXECUTION-QUALITY IMPROVEMENT WORTH A FUTURE VERSIONED DEPLOYMENT REVIEW, at CNY 50k
and above only.** The candidate fails its own pre-registered cost and turnover bounds at CNY 10k
and 30k. Nothing is deployed by this study.

## First, a correction to the claim that motivated it

A previous review described small-account residual cash as a **ratchet** — cash accumulating
because it is never redeployed. Measured over 132-136 consecutive-rebalance changes per tier,
strictly before the 2026-06-18 inception, that word was wrong:

| tier | cash rises | cash falls | mean change | lag-1 autocorr | longest monotone rise |
|---|---|---|---|---|---|
| 10,000 | 52.3% | 47.7% | +0.026% | 0.147 | 5 |
| 30,000 | 52.6% | 47.4% | +0.006% | 0.145 | 5 |
| 100,000 | 51.5% | 48.5% | +0.010% | 0.116 | 4 |
| 5,000,000 | 51.5% | 48.5% | +0.0004% | 0.223 | 4 |

The cash ratio is **mean-reverting noise, not a ratchet**. Three consecutive rises in the forward
quarter sit inside a historical distribution whose longest monotone run is 4-5. What is real is the
*level*, and its decomposition.

## Two channels, and only one of them is avoidable

The production allocator floors each name's target independently, so leftovers are never pooled:

* **REMAINDER** — the slice buys k lots and leaves less than one more. Scales like 1/capital.
  Historical median: 15.06% of equity at 10k, 5.79% at 30k, 1.73% at 100k, 0.03% at 5M.
* **ZERO-SLICE** — one lot costs more than the whole equal-weight slice, so the name gets nothing
  and its entire slice sits in cash. A step function of the basket's price level, not of capital.
  Historical median: **0.00% at every tier** — historically almost never binding.

The 2026 forward quarter was unusual in the second channel, not the first. Top-10 basket prices at
the four D=1 executions, with the count of names whose one lot exceeds a 10k account's slice:

| execution | top-10 price range | unaffordable at 10k | at 30k | at 100k |
|---|---|---|---|---|
| 2026-06-18 | 3.08 - 57.55 | 3/10 | 1/10 | 0/10 |
| 2026-07-01 | 2.93 - 29.99 | 3/10 | 1/10 | 0/10 |
| 2026-08-03 | 3.20 - **184.09** | 4/10 | 2/10 | **1/10** |
| 2026-09-01 | 3.03 - 61.50 | 5/10 | 3/10 | 0/10 |

So the 62% was 30-50% of equity locked in slices that could not buy one lot, on top of ordinary
remainder — driven by the basket drifting toward expensive names, including one whose single lot
(CNY 18,409) exceeded even a 100k account's slice.

A related documented claim was checked and **stands**: `docs/paper_trading.md` says "at CNY 10k the
book is nearly full (9.6/10)". Historically that is right (mean 9.68 names, all ten on 72.2% of
rebalances). The forward quarter's 6.24 is the outlier, not the documentation.

## The candidate

One candidate, frozen in advance. Of all integer-lot books affordable after real fees, take the one
closest to the intended weight vector:

```
f(k) = sum_i (a_i - t_i)^2 + (a_cash - t_cash)^2
```

weights **valued** at the execution bar's close, budget **charged** at the slipped price plus
commission (CNY 5 minimum), stamp tax and transfer fee. Tie-breaks, frozen: objective, then
residual cash, then incremental cost, then turnover, then ticker order.

### floor/ceil is NOT a sufficient search space — proof by counter-example

The pre-registration refused to assume the optimum lies in `{floor(x_i), ceil(x_i)}`. It does not:

| tier | rebalances whose optimum left the floor/ceil box |
|---|---|
| 10,000 | 26/137 (19.0%) |
| 30,000 | 13/137 (9.5%) |
| 50,000 | 11/137 (8.0%) |
| 100,000 | 26/137 (19.0%) |
| 500,000 | 46/137 (33.6%) |
| 1,000,000 | 71/137 (51.8%) |
| 5,000,000 | 129/137 (94.2%) |

The share RISES with capital, which is the opposite of the intuition that only cramped small
accounts need a wide search: at a large account the ideal lot count is big, one lot is a tiny
weight, and the optimum routinely sits two or more lots above the floor on several names at once
while it mops up the last of the cash.

The mechanism is in the marginal algebra: adding a lot to name `i` changes the objective by
`2d^2 + 2d[(a_i - t_i) - (a_cash - t_cash)]`, which is still negative when `a_i` is above target
provided cash is further above target — exactly a small account holding an expensive name.

### The optimiser had to be fixed, and the fix is recorded

Greedy-from-all-floor plus local search was **not** good enough. Against an independently derived
parametric-frontier solver — the objective separates under a multiplier into
`k_i = round((t_i - lam)/c_i)`, so sweeping `lam` across its breakpoints enumerates the frontier
with no search heuristic — the greedy was strictly worse on **184 of 280** rebalances. The shipped
solver runs both starts and keeps the better under the unchanged frozen key. After the fix the
frontier solver is strictly better in **0** cases, identical in 218/280, and the local search
strictly better in 62.

A second self-inflicted defect is also recorded: a speed optimisation that *skipped* over-budget
frontier points instead of repairing them silently destroyed the best small-account books — on the
canonical toy case it turned `[500, 200, 100]` (objective 0.114) into `[1400, 400, 0]` (0.150),
i.e. it stopped buying the expensive name entirely. Repair is now budgeted, not skipped.

## Baseline parity

`allocator=None` reproduces the pre-change engine **bit-identically** in all seven tiers, verified
against the canonical paper ledger that the pre-change code wrote (`d_equity = 0.000e+00`,
positions equal, every tier). The candidate is opt-in; the production path is untouched.

## Primary results — allocation quality, controlled

Both allocators scored from the **identical pre-trade book** at all 137 historical rebalances, so
path divergence cannot contaminate the comparison.

| tier | TE base | TE cwil | cash base | cash cwil | max dev base | max dev cwil | all-10 base | all-10 cwil |
|---|---|---|---|---|---|---|---|---|
| 10,000 | 0.1890 | **0.0443** | 17.53% | **1.04%** | 4.50% | 2.75% | 72.3% | 86.9% |
| 30,000 | 0.0631 | **0.0130** | 5.82% | **0.30%** | 1.54% | 0.81% | 97.8% | 100.0% |
| 50,000 | 0.0389 | **0.0078** | 3.60% | **0.18%** | 0.94% | 0.46% | 100.0% | 100.0% |
| 100,000 | 0.0195 | **0.0040** | 1.83% | **0.11%** | 0.48% | 0.23% | 100.0% | 100.0% |
| 500,000 | 0.0043 | **0.0010** | 0.40% | **0.05%** | 0.10% | 0.05% | 100.0% | 100.0% |
| 1,000,000 | 0.0024 | **0.0007** | 0.22% | **0.05%** | 0.05% | 0.03% | 100.0% | 100.0% |
| 5,000,000 | 0.0009 | **0.0005** | 0.09% | **0.04%** | 0.01% | 0.01% | 100.0% | 100.0% |

Tracking error improves 2-4x at every tier, and the maximum single-name deviation **also** improves
— the candidate is closer to equal weight on both the cash axis and the concentration axis. The two
allocators produce an identical book on 0.0% of rebalances, so this is not a rounding curiosity.

## The frozen acceptance bounds split by capital

| tier | TE med | TE p90 | cash | names | cost | turnover | max dev | over-2x | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 10,000 | ok | ok | ok | ok | **+18.62 bp FAIL** | **+50.7% FAIL** | -1.75pp ok | 1.5% ok | **FAIL** |
| 30,000 | ok | ok | ok | ok | **+3.21 bp FAIL** | +9.5% ok | -0.73pp ok | 0.0% ok | **FAIL** |
| 50,000 | ok | ok | ok | ok | +1.41 bp ok | +6.5% ok | -0.48pp ok | 0.0% ok | PASS |
| 100,000 | ok | ok | ok | ok | +0.09 bp ok | +0.8% ok | -0.25pp ok | 0.0% ok | PASS |
| 500,000 | ok | ok | ok | ok | +0.04 bp ok | +0.3% ok | -0.05pp ok | 0.0% ok | PASS |
| 1,000,000 | ok | ok | ok | ok | +0.03 bp ok | +0.5% ok | -0.02pp ok | 0.0% ok | PASS |
| 5,000,000 | ok | ok | ok | ok | +0.00 bp ok | +0.0% ok | -0.00pp ok | 0.0% ok | PASS |

**Not all seven tiers pass, so the pre-registered blanket acceptance condition is not met.** The
30k failure is by 0.21 bp against a 3.00 bp limit; it is reported as a failure regardless, because
a bound relaxed after seeing the number is not a bound.

The economics of the failure are clear rather than mysterious: putting the last of the cash to work
at a small account means extra one-lot trades, and a one-lot trade of a few hundred CNY pays the
CNY 5 minimum commission — 50-150 bp on that trade alone.

## Falsification — where it hurts

* Tracking error is **never** worse at any tier except 5M, where it is worse on **7.3%** of
  rebalances. At 5M the baseline is already near-perfect (median TE 0.0009) and the two are
  effectively tied; that 7.3% is the honest residual.
* Cash utilisation is never worse below 5M; worse on 7.3% of rebalances at 5M.
* Of the rebalances where the baseline already had near-zero cash (119 at 500k, 135 at 1M, 137 at
  5M), the candidate made **0.0%** worse at 500k/1M and 7.3% worse at 5M.
* Concentration: the candidate's **p99** maximum single-name deviation is 10.47% at the 10k tier
  against a 10% target weight — i.e. once in a hundred rebalances a 10k account puts roughly a
  fifth of the book in one name. It exceeds twice the target weight on 1.5% of rebalances, inside
  the frozen 5% allowance, but this is the concentration cost of redistributing an unaffordable
  name's slice and it is real at the smallest tier only.
* High-price and low-price baskets both improve, by almost the same amount (median TE gain at 10k:
  +0.1367 high-price vs +0.1331 low-price), so the gain is not confined to the expensive-basket case
  that motivated the study.

## Secondary diagnostics — cannot decide the verdict, and one of them is unflattering

| tier | CAGR base | CAGR cwil | maxDD base | maxDD cwil | costs base | costs cwil |
|---|---|---|---|---|---|---|
| 10,000 | 7.130% | 8.745% | -28.52% | **-33.05%** | 3,602 | 4,661 |
| 30,000 | 9.095% | 9.241% | -30.72% | **-33.13%** | 6,650 | 7,144 |
| 100,000 | 9.717% | 9.958% | -32.62% | -33.09% | 14,686 | 15,093 |
| 5,000,000 | 10.153% | 10.157% | -33.08% | -33.10% | 574,237 | 574,707 |

Return is higher everywhere, and that is **not** why the candidate is being considered. The
informative column is drawdown: it gets **worse** at every tier, most at 10k (-28.5% to -33.1%).
That is the correct reading of the whole study — the small account's shallower historical drawdown
was never risk control, it was 17% of the book sitting in cash. Executing the intended portfolio
faithfully means also taking the intended risk. A user who liked the 10k drawdown was liking an
accident.

## Capital implementability, as a curve

Median post-rebalance cash and the share of rebalances holding all ten names, both allocators:

| tier | cash (base) | cash (cwil) | all-10 (base) | all-10 (cwil) |
|---|---|---|---|---|
| 10,000 | 17.53% | 1.04% | 72.3% | 86.9% |
| 30,000 | 5.82% | 0.30% | 97.8% | 100.0% |
| 50,000 | 3.60% | 0.18% | 100.0% | 100.0% |
| 100,000 | 1.83% | 0.11% | 100.0% | 100.0% |
| 500,000+ | <0.40% | <0.05% | 100.0% | 100.0% |

The natural inflection in the baseline curve is between 10k and 30k: all-ten jumps 72.3% -> 97.8%
and median cash falls 17.5% -> 5.8%. That **agrees with the existing documented threshold**
(`live/strategy.py` and `docs/paper_trading.md`: start at >= CNY 30k, 10k infeasible, 50k
comfortably viable) and this study does not redefine it. The candidate would lower the faithful-
execution floor — at 30k it reaches 100% all-ten and 0.30% cash — but it fails its cost bound at
exactly those two tiers, so the practical floor does not move.

## Governance

No production change is authorised by this study. `DEPLOYED`, `PAPER_INCEPTION`, the canonical D=1
ledger, the D=5 candidate, issue #22's frozen definition and the scheduled task are all untouched;
the engine's default path is byte-identical. No paper history is backfilled and nothing is
presented as though the candidate had been in use since 2026-06-18. Any eventual adoption must
begin from an explicitly recorded future decision date.

`n_hold` was deliberately NOT studied here — changing the basket size and the allocator together
would confound two questions. It may become its own pre-registration now that this one is closed.

## Reproducing

```
conda activate hermes
python scripts/lot_residual_diagnostic.py     # baseline-only mechanism, ~2 min
python scripts/lot_allocation_study.py        # the frozen battery, ~12 min
pytest tests/test_lots.py -q                  # 22 tests
```
