# Multi-factor diversification (option B): a modest reversal tilt that works

After A1 (market-timing) and A2 (position sizing) failed to cut plain value's **-33% max
drawdown**, B asks whether *breadth* -- combining value with a diversifying factor -- helps.
**It does, for risk-adjusted return: a light reversal tilt raises net Calmar from 0.28 to
~0.32, above the A2 inverse-vol win (0.30).** But the gain is in the numerator (return), not
the drawdown -- see the caveat. This result corrects an earlier under-sampled version that
wrongly concluded "nothing helps"; that mistake, and how it was caught, are recorded below.

All results: survivorship-free PIT HS300, A-share frictions, top-10 monthly, 2015-2025.
Reproduce: `python scripts/multifactor_demo.py`.

## 1. Diagnosis — a screen, not a proof

Monthly rank-IC **correlation** across candidate factors (low/negative = bets diverge):

|        | ep    | bp    | sp    | lowvol | rev1m | mom6_1 |
|--------|------:|------:|------:|-------:|------:|-------:|
| ep     | 1.00  | 0.86  | 0.84  | 0.81   | -0.17 | 0.05   |
| bp     | 0.86  | 1.00  | 0.85  | 0.78   | -0.16 | -0.06  |
| sp     | 0.84  | 0.85  | 1.00  | 0.71   | -0.18 | -0.06  |
| lowvol | 0.81  | 0.78  | 0.71  | 1.00   | -0.31 | 0.12   |
| rev1m  | -0.17 | -0.16 | -0.18 | -0.31  | 1.00  | -0.27  |
| mom6_1 | 0.05  | -0.06 | 0.04  | 0.12   | -0.27 | 1.00   |

The value family (ep/bp/sp) and low-vol are mutually 0.71-0.86 correlated -- **redundant**
(this is *why* the A2 value×low-vol blend added nothing). Reversal is the lone negative.
But treat the matrix as a **screen only**: it is window-sensitive and lives in rank space,
not return space, so it mis-ranks the diversifiers -- a 10-day reversal shows ≈0 IC-corr yet
combines *best* (below). The real evidence is the combination curve, not this table.

## 2. Combination — sweep the value:reversal mix densely (NET and GROSS)

The earlier error was sampling only 50/50, 67/33, 75/25 -- all heavy-reversal, all in the
worst region. The full curve at 1M (NET = A-share frictions; GROSS = zero-cost, to test
whether any benefit is real alpha or just survives costs):

| variant (1M)          | CAGR  | maxDD  | net Calmar | gross Calmar | costs    |
|-----------------------|------:|-------:|-----------:|-------------:|---------:|
| value (base)          | +9.5% | -33.6% | 0.28       | 0.30         | 71,502   |
| value/inverse-vol (A2)| +10.1%| -33.1% | 0.30       | 0.32         | 79,835   |
| val+rev1 1/1 (50/50)  | +9.2% | -37.9% | 0.24       | 0.29         | 298,803  |
| val+rev1 2/1          | +9.8% | -33.5% | 0.29       | 0.32         | 185,799  |
| val+rev1 3/1          | +9.6% | -33.5% | 0.29       | 0.31         | 140,951  |
| **val+rev1 4/1**      | +10.3%| -33.1% | **0.31**   | 0.33         | 125,251  |
| **val+rev1 5/1**      | +10.5%| -33.0% | **0.32**   | 0.34         | 112,988  |
| **val+rev1 7/1**      | +10.5%| -32.9% | **0.32**   | 0.34         | 103,466  |
| val+rev1 9/1          | +10.1%| -32.8% | 0.31       | 0.32         | 96,339   |
| val+rev10 3/1 (10-day)| +11.9%| -33.6% | 0.35       | 0.38         | 159,502  |

Two things matter. First, a **light** reversal tilt (value ~80-90%) beats both the value
baseline and A2 -- and the **4/1–9/1 region is a plateau (0.31-0.32), not a lucky spike**.
Heavy reversal (50/50) is *worse* (0.24): too much of a high-turnover, low-return leg.
Second, **GROSS Calmar rises too** (0.30 → 0.34), so the gain is genuine pre-cost alpha, not
luck surviving frictions -- directly the opposite of what the earlier version claimed.

A 10-day reversal is stronger still (net 0.35 / gross 0.38), but picking the exact
window/weight on the same data is in-sample overfitting. **Take val+rev1 5/1–7/1 (net 0.32)
as the robust, deployable read;** treat the 10-day result as directional, not a target.

## 3. Robustness across capital tiers

| tier       | value | rev1 5/1 | rev1 7/1 | rev10 3/1 |
|------------|------:|---------:|---------:|----------:|
| 100,000    | 0.28  | 0.31     | 0.31     | 0.34      |
| 1,000,000  | 0.28  | 0.32     | 0.32     | 0.35      |
| 10,000,000 | 0.28  | 0.32     | 0.32     | 0.36      |

The tilt's edge holds at every tier (it is not a large-account-only artifact).

## 4. Turnover buffer — still the wrong tool here

A rebalance buffer (`rebalance_band`, hysteresis) cuts churn but **hurts** the tilt just as
it hurts plain value: val+rev1 5/1 Calmar 0.32 (b0) → 0.27 (b5) → 0.21 (b20). Verified at
zero cost too, so it is a genuine timing give-up (value-driven strategies must rotate into
the cheapest names; hysteresis strands capital in names that became dear), not a cost or
lot-rounding effect. The mechanism is retained (correct, off by default) for genuinely
high-turnover signals; it is the application to value/reversal that fails.

## Quality + broad diversification on HS300: cures concentration, costs return (rejected here)

A natural worry: the value top-10 is ~70% banks (HHI 0.60), so it is an unintended single-sector
bet. Does adding a **quality factor (ROE)** and **diversifying across factors** fix that? ROE is
reconstructed free from the lake as `pbMRQ/peTTM` (E/B; validated 600519≈35%, 601398≈10%). Swept on
HS300, top-10 monthly, 1M, net & gross, measuring sector concentration (`scripts/quality_multifactor_demo.py`):

| config                    | net Calmar | gross Calmar | maxDD  | sector HHI | max sector |
|---------------------------|-----------:|-------------:|-------:|-----------:|-----------:|
| value (base)              | 0.26       | 0.27         | -33.6% | 0.60       | 71% |
| **value + rev 5/1 (deployed)** | **0.30** | **0.32**  | -33.0% | 0.58       | 70% |
| value + quality 2/1       | 0.11       | 0.12         | -46.1% | 0.43       | 57% |
| value + quality 1/1       | 0.03       | 0.03         | -64.9% | 0.26       | 39% |
| val + qual + rev 3/2/1    | 0.12       | 0.16         | -38.3% | 0.33       | 46% |
| 5-factor equal (ep,roe,rev,mom,lowvol) | 0.20 | 0.25    | -41.2% | 0.37       | 51% |

**Diversification works on concentration (HHI 0.60 → 0.33, max-sector 71% → 46%) but is
counterproductive on every return/risk axis** -- net AND gross Calmar fall (0.30 → 0.11-0.20) and
maxDD *deepens* (-33% → -38% to -65%). The gross collapse means it is genuine signal degradation,
not a friction artifact. This is the **third independent confirmation** (after A3 sector-neutral,
A4 size) that **in HS300 the cheap-bank concentration is load-bearing** -- it is the source of both
the return and the residual defensiveness. ROE in HS300 tilts toward expensive growth (酒/科技 at
high multiples) that drew down harder over 2015-2026. **Verdict: keep the deployed HS300 config at
value + light reversal; do NOT add quality / broad diversification / sector-spread here.**

**Crucial scope note:** this null is SPECIFIC to HS300 (a large-cap, financials-heavy index where
value ≈ banks). On a **wider CSI500 universe**, value is not a bank bet and there is real
cross-sector breadth to harvest -- so quality (quality-value avoids cheap-for-a-reason traps),
factor diversification, and sector-relative selection should be RE-EVALUATED there, and are the
substance of the CSI500 line. The improvement to the *main* approach is the **universe**, not
factor-stuffing within HS300.

## Conclusion

**B succeeds on risk-adjusted return:** a modest reversal tilt (value ~85% / reversal ~15%)
lifts net Calmar 0.28 → ~0.32 (gross 0.30 → 0.34), above A1/A2, robustly across the weight
plateau and all capital tiers. Deployable configuration: **value + a light 1-month-reversal
tilt (≈5/1)**; inverse-vol does not stack on top (reversal already supplies the
diversification) and the turnover buffer should stay off.

**But the -33% drawdown is NOT cured.** Every variant sits at maxDD -32.8% to -33.8%; the
Calmar gain is entirely CAGR (~+1pp). The systematic, whole-market drawdown is unchanged --
consistent with A1 and A2. The two selection-side levers this flagged as "next" were then
tested and **both FAILED**: **sector/industry neutralization** (A3) and a **size tilt** (A4)
each *deepen* the drawdown monotonically (see [docs/risk_control.md](risk_control.md)) -- value
in HS300 IS the bank trade, and that concentration is the strategy's residual *defensiveness*.
The only equity lever left is a **wider universe (CSI 500** -- free + PIT via BaoStock; CSI
1000 is not in BaoStock**)**, rated low-odds and gated on modeling 涨跌停 no-fill first. Truly
cutting −33% (not re-sampling the same beta) realistically needs a **hedge overlay** -- a new
scope. The base strategy accepts −33% as understood systematic market risk.

## Key finding: the sampling mistake this caught

The first version of this study tested only three heavy-reversal weights (50/50, 67/33,
75/25), which happen to bracket the worst region of the curve, and concluded "no combination
beats value; reversal's benefit is fully eaten by frictions." Adversarial review (a
claim-skeptic agent) ran a denser sweep on the *same* engine and refuted it: light tilts
beat the benchmarks, and the gross (zero-cost) curve rises -- so the benefit is real alpha,
not a friction story. Lesson: **sweep the whole parameter curve before declaring a null;** a
sparse sample can land entirely in the wrong region and manufacture a false negative. (The
opposite trap -- cherry-picking the single best point, e.g. rev10 3/1 = 0.35 -- is handled
by reporting the plateau and the cross-tier table, and deploying the conservative read.)

### Fix shipped with this study

`factors.library.restrict_to_universe` is now applied in `model.walk_forward.build_dataset`
too -- it had the same standardize-over-the-union survivorship leak A2 surfaced. The ML
results were never the headline, but the leak is now closed everywhere standardization runs.
