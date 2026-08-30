# Rebalance timing: does the calendar day matter? — REJECTED / NOT IDENTIFIED

Pre-registered as [issue #21](https://github.com/Rhymer-Lcy/hermes-quant/issues/21) before any
sweep ran. Script `scripts/rebalance_timing_study.py`; schedule helper
`src/hermes/research/backtest/schedule.py`; tests `tests/test_schedule.py` and
`tests/test_rebalance_timing_gates.py`; figure `results/figures/rebalance_timing.png`.

A7 settled the *frequency* (monthly beats weekly and quarterly at every tier). This asks the
orthogonal question: conditional on monthly rebalancing, does the within-month calendar day carry a
stable effect? All 31 days were enumerated and the study was framed as a **multiple-testing
problem** — enumerating 31 candidates over a finite sample guarantees a maximum, and the maximum is
not evidence.

## Parity gate — passed bit-for-bit

D=1 must reproduce the canonical month-end engine or the study is meaningless. It does, exactly:

| | canonical engine | D=1 schedule |
|---|---|---|
| equity | — | max abs diff **0.0000000000e+00** |
| rebalances | 131 | 131 |
| trades | 1,547 | 1,547 |
| total costs | 112,987.575651 | 112,987.575651 |
| CAGR / maxDD | +10.4922% / −33.0403% | identical |

The cost figure also matches the **112,988** published in `docs/multi_factor.md`, so the D=1 path
reproduces not merely itself but the already-documented deployed baseline.

Engine change: one opt-in `schedule=` argument. `schedule=None` is byte-identical to before, and
`rebalance_freq="M"` semantics are untouched. Entries are filtered to `0 <= signal < exec < n`, so
a malformed schedule degrades to fewer rebalances and can never introduce look-ahead.

## The headline, and why it is not the answer

**Historical CAGR maximum: D=5, +12.67% against D=1's +10.49% — a gap of +2.18 pp per year.**
D=3–9 *all* beat the baseline; D=11–31 all lose to it, deteriorating steadily toward month-end.

| D | CAGR | ΔCAGR | Calmar | maxDD | costs |
|---|---:|---:|---:|---:|---:|
| 1 (baseline) | +10.49% | — | 0.318 | −33.04% | 112,988 |
| 3 | +11.65% | +1.15% | 0.356 | −32.67% | 125,423 |
| 4 | +12.03% | +1.54% | 0.354 | −34.00% | 131,190 |
| **5** | **+12.67%** | **+2.18%** | **0.378** | −33.52% | 141,005 |
| 6 | +12.47% | +1.97% | 0.369 | −33.81% | 139,972 |
| 9 | +12.03% | +1.53% | 0.360 | −33.38% | 133,294 |
| 20 | +7.66% | −2.83% | 0.198 | −38.71% | 100,215 |
| 28 | +6.72% | −3.77% | 0.196 | −34.33% | 93,764 |

(All 31 rows, both net and gross, are in `results/backtests/rebalance_timing_*.csv`; nothing is
truncated.)

Two things are worth stating before the verdict, because both cut *for* the effect:

- **It is not a cost artefact.** The winning days trade *more*, not less — D=5 pays 141,005 in
  costs against D=1's 112,988 — and still wins. The gross (zero-cost) sweep shows the same shape
  (D=5 +13.36% vs D=1 +11.12%), so gate B passes comfortably.
- **The shape is smooth and monotone**, not a lone spike: a 7-day contiguous run (D=3–9) beats the
  baseline on CAGR and an 8-day run (D=3–10) on Calmar. Gate C passes.

## Verdict — REJECTED / NOT IDENTIFIED

Every gate is evaluated **per candidate day**, so the eight-gate conjunction certifies one day
rather than a disjunction across days (see "Implementation audit" below — the first version got
this wrong).

| D | A Holm | B gross | C plateau | D regimes | E decade | F risk-adj | G drawdown | H tiers | passed |
|---|---|---|---|---|---|---|---|---|---:|
| 3 | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | 6/8 |
| 4 | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | 6/8 |
| **5** | **✗** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **7/8** |
| 6 | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | 5/8 |
| 7 | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | 6/8 |
| 9 | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | 5/8 |

**No day passes all eight. The best is D=5 at 7 of 8, failing exactly the gate that guards against
this study's own design: Holm-corrected significance.**

### Why D=5 fails, in three independent ways

**1. The monthly evidence is weak, and 31 comparisons make it weaker.** D=5's paired monthly
difference is +0.1497%/month with Newey-West HAC t = **+1.508** — p_raw 0.1316, *insignificant
before any correction at all*, and p_holm 1.0000 after. A +0.15%/month mean compounds into a
visible +2.18pp CAGR gap over 11 years while remaining statistically indistinguishable from zero.
That gap between "looks big on a curve" and "is measurable" is the whole reason the gate exists.

**2. One year carries it.** On the complete-decade window 2016–2025 — dropping only 2015 — D=5
collapses from +2.18% to **+0.06%**, and D=6/7/8/9 all turn negative. Only D=3 (+0.74%) and D=4
(+1.01%) survive, and neither is significant either.

**3. The ranking is unstable across regimes.** Mean Spearman ρ over *disjoint* sub-period pairs is
**+0.306**, and 2015-2018 vs 2019-2021 is **negative (−0.156)** — the ordering of the 31 days
partially *reverses* between the first two regimes. Regime-by-regime, most of the winning band
fails: in 2019-2021, D=3 is −1.50%, D=4 −0.96%, D=6 −1.57%, D=8 −0.94%, D=9 −0.11%. Only D=5
(+0.20%) and D=7 (+0.24%) stay positive, and barely.

**4. The champion wanders.** In a 2,000-draw year-block bootstrap, D=5 is the winner in only
**30.7%** of resamples, with 12 distinct winners (D=6 27.2%, D=9 12.2%, D=3 11.2%, D=4 10.1%).
By the pre-registered criterion this is **winner instability**.

## The finding that *is* solid: the losing side

The one comparison surviving Holm correction is **D=29, and it is significantly WORSE**:
−0.2741%/month, HAC t = **−3.294**, p_holm 0.0296. More broadly, the only days differing from D=1
at even an uncorrected p ≤ 0.05 are **D=26, 27, 28, 29, 30, 31 — every one of them negative**.

**The evidence that late-month rebalancing hurts is far stronger than the evidence that early-month
rebalancing helps.** That asymmetry is the study's most reliable output, and it is the third time
this repository has found the trustworthy side of an effect to be the losing one (see
`docs/preannounce_study.md`).

## Operational flexibility — the question behind the question

- **D=2 through D=11 are all statistically indistinguishable from D=1** (raw p > 0.05, before any
  multiplicity correction). The early-month region is one flat plateau as far as this sample can
  resolve.
- **95.9% of bootstrap draws pick a winner inside D=3–9** — so while no single day is certifiable,
  the *region* is stable. The day wanders; the neighbourhood does not.
- **D ≥ 26 is measurably worse** and should be avoided, on the strongest statistical footing in the
  study.

So: **monthly frequency matters; the exact early-month day does not.** Operational flexibility
within roughly the first ten calendar days is licensed by the data. Slipping a rebalance by a few
days for operational reasons costs nothing measurable; slipping it to month-end does.

Note that D=1 sits at the *left edge* rather than the centre of the favourable band — D=2 is
−0.36% and D=3–9 are all positive. That is an observation, not a recommendation: none of it
survives the gates.

## Secondary diagnostic (no verdict weight)

Executing on the **Nth trading day** instead of the Nth calendar day reproduces the same shape —
the 3rd trading day gives +2.00%, and the curve decays through the middle of the month. The effect
therefore looks like a **position-in-month** phenomenon rather than anything about calendar
numbers, which is what one would expect if it is related to the turn-of-month/liquidity cycle
rather than to the digits on a date.

## Implementation audit — two critical defects found before the sweep ran

An adversarial audit (5 independent reviewers, findings then attacked by 2 skeptics each; 35 agents
total) was run against the implementation **before** any result was produced. Four findings survived
refutation, two of them critical, and all four are fixed in the committed code:

1. **(critical) Gate A tested the wrong thing.** `A = paired["reject"].any()` asked "is *some* day
   significant?" while gates C–H were read at the CAGR champion. The conjunction could therefore
   certify a *disjunction*: the auditor demonstrated a fabricated case printing CONFIRMED for a
   champion whose own Holm p was 1.0. Gates are now evaluated per candidate day.
   `tests/test_rebalance_timing_gates.py::test_significance_from_another_day_cannot_confirm_the_champion`
   reproduces that exact counter-example.
2. **(critical) Gate B had the same defect** and additionally never read the `gross` sweep passed
   into it — a dead parameter where the cost-artefact test should have been.
3. **(major) The rank-stability table correlated nested windows** (2015-2018 sits inside "first
   half"; 2022-2025 inside "second half"), where ρ is inflated by construction, and printed a
   Spearman p that assumes independent observations when adjacent D share most of their execution
   calendar. Now restricted to disjoint pairs with the invalid p removed.
4. **(self-found) The secondary window computed the signal after slicing**, leaving the 20-day
   reversal leg NaN over the window's first 20 bars (429 signal cells). Now computed on the full
   panel then sliced — still strictly point-in-time.

## Recommendation

**Keep D=1.** It is inside a broad flat region, it is the simplest and most auditable rule, it is
already automated, and nothing in 31 enumerations beats it on evidence rather than on hindsight.

No forward-test candidate is opened. Gate A failed *before* multiplicity correction, so there is
nothing here worth spending future data on. Were that to change, the candidate would be the
**region D=3–9**, never a single day — the bootstrap is explicit that the region is the stable
object and the day is not.

`DEPLOYED` is unchanged, and the D=1 paper ledger was neither touched nor rewritten.

## Reproduce

```
python scripts/rebalance_timing_study.py     # ~3 minutes; parity gate asserts before any sweep
pytest tests/test_schedule.py tests/test_rebalance_timing_gates.py
```
