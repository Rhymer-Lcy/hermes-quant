# Out-of-sample / edge-decay gate: does the HS300 value+reversal edge survive?

The deployed book (value = earnings yield, plus a light 1-month-reversal tilt at 5:1, equal-weight
top-10, monthly) reports a full-sample net Calmar ~0.32 over 2015-2025 (see
[multi_factor.md](multi_factor.md)). A full-sample number cannot distinguish a stable edge from
one that was real early and has since been arbitraged away. The sister project **plutus-quant**
(US equities) learned this directly: a cross-sectional signal that looked strong over 2009-2024 in
aggregate was shown, by a trailing-window decomposition, to have faded to a statistical zero after
~2020 and to print a negative holdout year. The decisive evidence there was the *multi-year fade*,
not any single year. This applies the identical discipline to hermes, on its own engine, its own
single-source deployed signal (`live.strategy.deployed_signal` — zero train/serve drift), and its
own A-share frictions.

Reproduce: `D:\Anaconda3\envs\hermes\python.exe scripts/oos_decay_study.py`.

## The hypothesis being tested (and why it might be true)

The A-share value/small-cap premium is commonly explained as a structural inefficiency of a
retail-dominated, hard-to-arbitrage, capital-controlled market (Liu-Stambaugh-Yuan CH-3/CH-4).
That structure has been eroding: MSCI A-share inclusion began 2018-06; registration-based IPO
arrived on STAR Market (2019-07), ChiNext (2020-08), and the main board (2023-02), deflating the
shell/IPO-scarcity premium; institutional and foreign capital have risen. If the edge is a
structural premium being competed away, its IC and risk-adjusted return should **fade
monotonically** across 2015-2018 → 2019-2021 → 2022-2025 — the plutus pattern.

## Result — the institutionalization-decay hypothesis is REJECTED; the edge is V-shaped, not fading

Monthly cross-sectional rank-IC of the deployed composite on the PIT HS300 universe
(132 non-overlapping monthly observations):

| window | n | mean IC | t | 95% CI | hit% |
|---|--:|--:|--:|---|--:|
| FULL 2015-2025 | 132 | +0.0563 | +3.11 | [+0.021, +0.092] | 59% |
| 2015-2018 pre/early | 48 | +0.0982 | +4.16 | [+0.051, +0.146] | 75% |
| 2019-2021 transition | 36 | −0.0056 | −0.15 | [−0.084, +0.073] | 44% |
| 2022-2025 post | 48 | +0.0608 | +1.91 | [−0.003, +0.125] | 54% |
| EARLY 2015-2020 | 72 | +0.0476 | +2.14 | [+0.003, +0.092] | 64% |
| HOLDOUT 2021-2025 | 60 | +0.0667 | +2.25 | [+0.007, +0.126] | 53% |

Per-year composite IC: 2015 +.072, 2016 +.144, 2017 +.136, 2018 +.040, **2019 −.039, 2020 −.068**,
2021 +.090, 2022 +.074, 2023 +.117, 2024 +.027, 2025 +.024.

The IC is **not** a monotone fade. It is V-shaped: strong 2015-2017, collapsed through the
**2019-2020 value/growth-bubble winter** (the only negative years — a global value drawdown, not an
A-share-specific institutionalization event), then recovered 2021-2023. The value (ep) leg carries
the same shape (2015-2018 IC +0.095 t3.43; 2019-2021 −0.015 t−0.35; 2022-2025 +0.061 t1.73);
reversal is a thin, regime-stable diversifier throughout (~+0.013 to +0.033, never significant
alone). This differs **in kind** from plutus's US signal, which faded smoothly to a statistical
zero and went negative out-of-sample. The A-share edge is regime-cyclical, not arbitraged away.

The holdout is the test plutus failed and this passes: **HOLDOUT 2021-2025 IC = +0.0667 (t=2.25)
is positive and statistically indistinguishable from the early period** (+0.0476): Welch
t = −0.52, p = 0.61; and P(holdout IC this weak | the full-sample edge were intact) = 0.65 — fully
consistent with the edge being intact.

## Calibration — what an adversarial audit walked back

A skeptic audit (tasked to refute, not confirm) reproduced every number, confirmed the pipeline is
leak-free and survivorship-clean (the signal at t recomputes bit-identically on data truncated to
≤ t; `members_asof` is strictly point-in-time; `restrict_to_universe` precedes standardization so
the union does not leak), and confirmed the inference is valid on non-overlapping, near-IID monthly
observations (IC lag-1 autocorrelation ≈ 0.02 full / 0.05 holdout — negligible). It corrected two
overstatements:

1. **The holdout's *standalone* significance is boundary-fragile.** It is not robustly significant;
   it rides on the window boundary and a few months:

   | recent window | composite t (p) | ep-only t (p) |
   |---|--:|--:|
   | 2020-2025 | +1.63 (0.107) | +1.40 (0.166) |
   | 2021-2025 | +2.25 (0.028) | +1.97 (0.053) |
   | 2022-2025 | +1.91 (0.063) | +1.73 (0.091) |

   Dropping the two largest IC months takes the 2021-2025 composite to t=1.81 (p=0.075). The
   defensible claim is therefore **"positive and not refuted — consistent with the early period"**,
   not "robustly significant on its own."

2. **The −33% drawdown floor is not cured; the recent regime is not benign.** An earlier version
   measured each sub-period's drawdown from a window-local high-water mark, reporting 2022-2025 at
   maxDD −15.6% / Calmar 1.01. That is a slicing artifact: the book *entered* 2022 already ~23%
   underwater (and bottomed at −30.1% within the window). Measured against the **carry-in**
   high-water mark (fixed), every regime carries the intrinsic value-style floor:

   | window | net CAGR | carry-in maxDD | net Calmar | net Sharpe |
   |---|--:|--:|--:|--:|
   | FULL 2015-2025 | +10.5% | −33.0% | 0.32 | 0.61 |
   | 2015-2018 | +8.7% | −33.0% | 0.26 | 0.48 |
   | 2019-2021 | +6.2% | −31.2% | 0.20 | 0.41 |
   | 2022-2025 | +15.9% | −30.1% | 0.53 | 0.96 |
   | HOLDOUT 2021-2025 | +10.9% | −31.2% | 0.35 | 0.70 |

   (Net @ ¥1,000,000, the fully-diversified regime, so the edge read is not contaminated by
   small-account lot/commission frictions.) The recent Calmar is ~0.35–0.53, not ~1.0. The −33%
   floor — shown in [risk_control.md](risk_control.md) A8 to be intrinsic value-style risk that no
   selection/weighting/universe/hedge lever removes — persists in every sub-period.

Two further honest qualifications:

3. **The raw premium has compressed.** The Q5-Q1 quintile spread (a pure-premium diagnostic; the
   short leg is *not* retail-tradeable in A-shares, so this is measurement, not a strategy) fell
   from +24.7%/yr (t=3.16) in 2015-2018 to +6.8%/yr (t=0.61) in 2022-2025. The edge is real but
   materially smaller than the full-sample headline, which is inflated by the spectacular 2015-2017.
4. **The last ~24 months drift toward zero.** Trailing 2024-2025 IC = +0.026 (t=0.56). This may be
   noise (2019-2020 looked similar before recovering) or early decay; the forward paper ledger
   (`live.paper`, running since 2026-06-22) is the instrument that will resolve it.

## Verdict

A-share value+reversal **passes the OOS/decay gate the US edge failed** — but in failure *mode*,
not in clean significance. It did **not** institutionalize away into a zero or negative holdout;
its weakness was a cyclical value winter (2019-2020) from which it recovered, and the recent
out-of-sample window is positive and indistinguishable from the in-sample period. That is a
genuinely stronger result than plutus's US conclusion ("disproven, do not deploy").

It is **not**, however, a proven permanent edge. The recent strength is modest (premium compressed
~3-4x from its peak), boundary-fragile in standalone significance, and carries an uncured −33%
value-style drawdown; the most recent two years are soft. The honest status is **on-watch and
structurally credible**: keep the deployed book, size for a −33% drawdown as understood intrinsic
risk, and let the forward paper ledger accumulate the real-time record that — unlike any backtest
slice — cannot be re-bounded after the fact. This is the same standard that retired the US edge;
A-shares clear it, the US did not, and the difference is structural (a retail-dominated,
under-arbitraged market harvesting a real value premium) rather than skill.
