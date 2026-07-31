# The bad-preannouncement avoidance overlay — CONFIRMED at the frozen bar, and operationally marginal

Pre-registered as [issue #20](https://github.com/Rhymer-Lcy/hermes-quant/issues/20) before any
portfolio number was computed. Script: `scripts/preannounce_avoid_study.py`. Follows
[#19](preannounce_study.md), which established the drift but could not price the overlay.

## The question an event study could not answer

#19 measured what a bad-news name does (−2.39% over 60d, t = −3.45; −4.18% over 120d, t = −5.08).
It could not measure what *avoiding* one is worth, because the value of an exclusion is **the loss
avoided minus the opportunity cost of the replacement that takes the slot**. A fixed-size book
always holds ten names; skipping one means buying the eleventh. Only a portfolio backtest running
the real selection, replacement and frictions settles that.

## Power, reported before any performance number

The pre-registration put the exclusion count first on purpose: the deployed book is HS300 value,
roughly 70% banks, whose earnings rarely swing hard enough to trigger a mandatory preannouncement.
"The rule never fired" and "the rule fired and did not help" are different findings.

| | |
|---|---:|
| negative-type events in the lake | 24,793 |
| events touching the HS300 price panel | 2,414 |
| **name-months the overlay actually changed** | **22** |
| frozen floor for INCONCLUSIVE | 20 |

Powered — but only just, by two name-months above a floor set in advance. That thinness governs
how much weight the result can carry, and it is stated here rather than buried.

## PRIMARY — CONFIRMED at the frozen bar

Frozen rule: net Calmar improves at the 1M tier **and** the sign is identical across all seven
capital tiers.

| capital | base CAGR / maxDD / Calmar | overlay | ΔCalmar | ΔCAGR | ΔmaxDD |
|---|---|---|---:|---:|---:|
| 10k | +7.4% / −28.5% / 0.26 | +7.4% / −28.5% / 0.26 | +0.0002 | +0.007% | 0.0000% |
| 30k | +9.5% / −30.7% / 0.31 | +9.9% / −30.7% / 0.32 | +0.0138 | +0.42% | 0.0000% |
| 50k | +9.8% / −32.2% / 0.30 | +10.2% / −32.2% / 0.32 | +0.0141 | +0.45% | 0.0000% |
| 100k | +10.1% / −32.6% / 0.31 | +10.6% / −32.6% / 0.32 | +0.0145 | +0.47% | 0.0000% |
| 500k | +10.5% / −33.0% / 0.32 | +10.9% / −33.0% / 0.33 | +0.0122 | +0.40% | 0.0000% |
| **1M** | **+10.5% / −33.0% / 0.32** | **+10.9% / −33.0% / 0.33** | **+0.0124** | **+0.41%** | **0.0000%** |
| 5M | +10.5% / −33.1% / 0.32 | +10.9% / −33.1% / 0.33 | +0.0127 | +0.42% | 0.0000% |

**CONFIRMED** — every tier improves. Two things must be read alongside it. The 10k tier improves by
0.0002, which is zero in all but arithmetic (that book holds ~2.8 names, so an exclusion rarely
alters it). And **ΔmaxDD is exactly 0.0000% at every tier**: the overlay adds return and does
nothing whatsoever for the −33% drawdown that motivated the search for an uncorrelated signal.

## Magnitude reconciliation — the test that mattered most

An effect far larger than its mechanism can produce is not that mechanism. The check:

| | |
|---|---:|
| #19 drift avoided per event | 2.39% |
| on an equal-weight 1/10 position | 0.239% of NAV |
| × 22 changes ÷ 11 years | **0.478% / yr explicable** |
| **observed** | **0.410% / yr** |
| ratio | **0.9×** |

The observed gain is *slightly smaller* than the mechanism's ceiling, which is what a real but
imperfectly-timed effect looks like. Had it come in at several times the ceiling, the result would
have been a few lucky episodes wearing the overlay's clothes.

## FALSIFICATION — the mirror

If avoidance helps, holding **only** the excluded names must hurt.

| book | CAGR | maxDD | Calmar | avg names |
|---|---:|---:|---:|---:|
| deployed | +10.5% | −33.0% | 0.32 | 9.9 |
| **mirror: bad-news names only** | **−3.5%** | **−68.1%** | **−0.05** | 8.6 |

Decisive. A book of names the overlay would have skipped loses money and drops 68%.

## Robustness grid — and the power that decides which cells count

The pre-registered power logic applies to every cell, not just the primary. Reporting a cell's
result without its exclusion count would repeat study B's sampling mistake in a new costume.

| cell | name-months | Calmar | reading |
|---|---:|---:|---|
| baseline (no overlay) | — | 0.32 | reference |
| 60d window | 17 | 0.33 | **underpowered** |
| **90d window [primary]** | **22** | **0.33** | powered |
| 180d window | 36 | 0.33 | powered |
| negative + 不确定 | 24 | 0.33 | powered |
| severe only (首亏/续亏/增亏) | **8** | 0.31 | **underpowered** |
| era 2015-2019 | 10 | Δ −0.00 | **underpowered** |
| era 2020-2026 | 12 | Δ +0.02 | **underpowered** |
| zero-cost gross | — | 0.34 → 0.35 | gain survives without costs |

**Every adequately powered cell agrees: 0.33 against a 0.32 baseline.** Every cell that disagrees
is underpowered.

Two apparent contradictions dissolve on inspection and are recorded because they looked damning:

- **"Severe-only makes it worse" (0.31)** fires 8 times. It is one or two events of noise, and
  there is a mechanical reason it barely fires at all: a name posting an outright loss has a
  negative or meaningless PE, so a 1/PE value ranking was never going to select it. The severe
  subset and the deployed book hardly intersect.
- **"The effect is only in 2020-2026"** rests on 10 versus 12 name-months. Neither era is powered;
  the split is not evidence of regime dependence in either direction.

The gain also survives with every fee and slippage zeroed (0.34 → 0.35), so it is not a cost
artefact.

## Verdict

**CONFIRMED at the frozen bar, and small.** +0.41pp of CAGR, +0.01 Calmar, **zero** drawdown
relief, from an overlay that alters the book roughly twice a year. The effect is real in the sense
that matters — it survives the tier sweep, the mirror, the zero-cost check, and it reconciles with
its own mechanism at 0.9× — but it does not address the problem that sent us looking. The −33%
drawdown is untouched, exactly as A1-A9 found for every other overlay.

**Deployment is a judgment call, and the case is genuinely balanced.** For: the mechanism is
understood, the evidence is internally consistent, and the paper pipeline already pulls vendor data
daily, so this is one more feed rather than new architecture. Against: +0.4pp/yr buys a new live
dependency and a new silent-failure mode (a stale or broken preannouncement feed either stops
excluding or excludes wrongly, and neither is loud), and the whole result stands on 22 name-months
— two above a floor set in advance.

If deployed, the fail-loud discipline already applied to the price feed must extend to this one:
a preannouncement pull below full coverage should fail the run rather than silently produce an
empty exclusion set.

## Reproduce

```
python scripts/build_preannounce_lake.py --resume    # the #19 lake, 44 periods
python scripts/preannounce_avoid_study.py
```
