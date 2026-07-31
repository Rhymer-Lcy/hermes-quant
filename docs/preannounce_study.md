# The mandatory earnings preannouncement (业绩预告) — REJECTED long-only, and the third sighting of the same wall

Pre-registered as [issue #19](https://github.com/Rhymer-Lcy/hermes-quant/issues/19) before any
event data was pulled. Script: `scripts/preannounce_study.py`; lake:
`scripts/build_preannounce_lake.py` → `data/parquet/preannounce.parquet`.

## Why this was worth testing

China mandates a preannouncement whenever a result crosses disclosure thresholds — a scheduled,
content-rich, cross-sectionally comparable disclosure event with no direct analogue in mature
markets. Two priors collided. In favour: post-earnings-announcement drift is the most durable
anomaly documented globally, and post-publication decay of anomalies is essentially a US
phenomenon — across 39 markets only the US shows reliable decline, plausibly because A-share
short-sale constraints keep arbitrage capital out. Against: 83.4% of 469 anomaly variables fail to
produce a significant raw spread in A-shares, and this lab's own base rate is ~17 pre-registered
rules with approximately zero harvestable alpha.

## Sample

44 reporting periods 2015-03-31 … 2025-12-31; 145,816 raw vendor rows; 73,251 deduplicated events
(1,533 later announcements excluded as revisions); 10,946 inside the PIT HS300+CSI500 universe from
2015; 10,628 with a tradable entry bar after the non-ST and ≥20-prior-bar filters.

Median disclosure lag is 13 days after period end (p10 −63, p90 +30) — the negative tail is the
interim-period habit of preannouncing before the period has closed.

## PRIMARY — REJECTED

Frozen rule: the top-quintile long leg's 60-day net abnormal mean must be POSITIVE with
monthly-clustered t > 2.

| horizon | n | top-quintile gross | net | monthly-clustered t |
|---|---:|---:|---:|---:|
| 20d | 2,040 | −0.43% | −0.63% | +0.13 |
| **60d** | **2,033** | **−1.25%** | **−1.45%** | **+0.81** |
| 120d | 1,990 | −1.76% | −1.96% | +1.28 |

The long leg is **negative at every horizon** and nowhere near the significance bar. **REJECTED.**
Per the pre-commitment, no window or threshold tuning follows.

## The spread is real — and it lives entirely on the side we cannot trade

A benchmark caveat first, stated because it changes how the levels should be read: **all**
preannouncers average −1.06% at 60d and −2.34% at 120d against the equal-weight universe, so the
level carries a common negative offset (preannouncers as a class underperform, and/or the naive
equal-weight benchmark is biased upward by forward-filled delisted names). The
**difference-in-means cancels any common offset** and is the clean read:

| horizon | positive − negative | monthly-clustered t | months |
|---|---:|---:|---:|
| 20d | +0.92% | +1.17 | 100 |
| 60d | **+3.15%** | **+2.50** | 99 |
| 120d | **+6.56%** | **+3.20** | 97 |

So the preannouncement **does** carry information. Decomposing it:

| news set | 60d mean | t | 120d mean | t |
|---|---:|---:|---:|---:|
| positive (预增/略增/扭亏/续盈) | −0.47% | +1.48 | −1.40% | +0.53 |
| **negative (预减/略减/首亏/续亏/增亏)** | **−2.39%** | **−3.45** | **−4.18%** | **−5.08** |

**The entire spread comes from bad news drifting down. Good news does not drift up.** A long-only
account can act on exactly none of it.

Note the sign disagreement in the positive rows: the pooled mean is negative while the
monthly-clustered t is positive. That is unequal month sizes — the January annual-preannouncement
crush dominates the pooled mean while contributing one observation to the monthly series. The
pre-registered inference unit is the month, so t = +1.48 is the read, and it is insignificant
either way.

## Robustness grid (all cells frozen in the issue, all reported)

- **Categorical instead of magnitude**: same picture — negative set significant (t = −3.45 at 60d,
  −5.08 at 120d), positive set not.
- **Magnitude rank within positive types only**: negative at every horizon, t between −0.55 and
  −1.05. Ranking good news by how good it is adds nothing.
- **Terciles instead of quintiles**: same shape, weaker.
- **Era split**: 2015-2019 top leg t = +1.33 at 60d; 2020-2026 t = −0.57. No era rescues it.
- **Annual vs interim**: neither significant.
- **Announcement earliness** — the one cell that is not a PEAD restatement, and the one that
  worked: the **latest** tercile drifts significantly down (60d −1.39% net, t = −2.50; 120d −2.60%,
  t = −3.51) while the earliest tercile is flat. "Good news early, bad news late" reproduces in
  A-shares — **and once again the tradable side is the short side.**

## CO-PRIMARY (frozen): correlation with the deployed strategy

101 overlapping months, correlation **+0.067** — the diversification gate (< 0.50) **passes**.
This is moot and reported only because the pre-registration binds it regardless of outcome: an
uncorrelated signal that does not make money is not a diversifier, it is noise.

## Verdict in context — the third sighting of the same wall

This is now the third independent finding in this repo with the identical shape:

| study | effect | significance | why it cannot be harvested |
|---|---|---|---|
| index reconstitution (#16) | fresh HS300 additions lag −3.4% | t = −3.09 | requires shorting |
| CSI500 native factors (A6) | low turnover-vol, IC +0.084 | t = 6.15 | a left-tail SHORT signal |
| **preannouncement (#19)** | **bad news drifts −4.18% / 120d** | **t = −5.08** | **requires shorting** |

The pattern is structural, not coincidental. In a retail-dominated market with binding short-sale
constraints, negative information stays unarbitraged precisely *because* nobody can trade it —
which is why it is measurable, and why it is measurable to no one's benefit. **A-share information
edges concentrate on the sell side; a long-only retail account is structurally on the wrong side of
the one thing that is reliably predictable.**

An avoidance overlay (do not hold names that just preannounced badly) is the only long-only
expression, and it is **not tested here** — it is a different hypothesis with a different
operationalization and needs its own pre-registration rather than a post-hoc read of this dataset.

## Reproduce

```
python scripts/build_preannounce_lake.py --resume    # 44 periods, ~40 min
python scripts/preannounce_study.py
```
