# Friends' rulesets, tested: eight pre-registered studies

A friend who trades A-shares stated their system orally (about half a year of live experience;
tested with their consent, no holdings shared). Every mechanically testable rule in it was
frozen into a pre-registered issue — [#2](https://github.com/Rhymer-Lcy/hermes-quant/issues/2)
quality-value entry, [#3](https://github.com/Rhymer-Lcy/hermes-quant/issues/3) inverted-pyramid
scale-in, [#4](https://github.com/Rhymer-Lcy/hermes-quant/issues/4) inverse-PE on cyclicals,
[#5](https://github.com/Rhymer-Lcy/hermes-quant/issues/5) the 120-day box,
[#6](https://github.com/Rhymer-Lcy/hermes-quant/issues/6) margin-balance timing, and later
[#7](https://github.com/Rhymer-Lcy/hermes-quant/issues/7) the Premier's-symposium attendance
signal (the friend's own proposal for quantifying the policy-selection layer) — BEFORE any
study code existed (for #7, before the event list existed). The friend's discretionary stock selection ("national policy 2–5 years out,
consumption upgrade") is untestable without a point-in-time watchlist and was declared out of
scope; verdicts bind the frozen operationalizations, not the person.

Shared conventions (the whole series): point-in-time HS300+CSI500 members, non-ST, ≥ 20 prior traded
days, 2015–present; signal at close t, execute at close t+1; proportional retail costs
(commission + transfer + slippage, sell-side stamp tax; the ¥5 minimum and 100-share lots are
sub-basis-point at study scale and excluded); verdicts on **monthly-clustered t** of daily
active returns. New shared data: annual-report ROE aligned by **publication date** (BaoStock
`pubDate` — a report counts only from the day it actually published), the CSRC industry
snapshot mapped to the friend's five buckets (frozen in issue #4's appendix; snapshot, not PIT
— disclosed), and the SSE margin-financing balance. Machinery in
`research/backtest/rule_portfolio.py` and `scale_in.py`, unit-tested first (the
fraction-drift model: between signals a held position's weight drifts with price — no free
daily rebalancing).

## #2 Quality-value entry — REJECTED (the quality gate is harmless; the PE gate is the damage)

`python scripts/quality_value_study.py` — 1,553 names; ~116/day pass quality, ~42/day pass both.

| variant | wealth | bench | active/yr | t(month) |
|---|---:|---:|---:|---:|
| **P_qv vs EW universe NET (the verdict)** | 0.893 | 1.517 | **−5.97%** | **−1.04** |
| P_q (quality only) vs universe | 1.718 | 1.517 | +0.86% | 0.34 |
| P_qv vs P_q (what the PE gate adds) | 0.893 | 1.718 | −6.83% | −1.38 |
| five-sector subset | 1.097 | 1.517 | −3.90% | −0.64 |
| 20%-per-bucket variant vs plain P_qv | 1.254 | 0.893 | +2.41% | 0.65 |

The friend's core buy rule ended at 0.893× while the equal-weight universe made 1.517× — an
absolute loss over 11.5 years. The attribution is one-sided: three years of ROE > 15% on its
own is harmless (+0.86%/yr, noise); requiring the name's OWN peTTM to sit at the bottom of its
five-year range is what subtracts (−6.83%/yr) — through this whole window that condition
selected quality names mid-derating, and the derating continued. The value-trap prior stated
in the issue is what materialized. The five-sector 20%-each overlay softens the damage but
still trails the universe; both eras negative; HS300 members (0.723) fared worse than
CSI500-only (1.179). Verdict needed t > 2; got **−1.04**. **REJECTED.**

## #3 Inverted pyramid vs lump-sum — REJECTED (wins 68% of events, loses the mean)

`python scripts/pyramid_entry_study.py` — 2,625 of 2,635 issue-#2 entries resolved; tranche
fills: one 22.5%, two 18.9%, all three 58.6%.

| sample | N | pyramid | lump | diff | t(month) | hit |
|---|---:|---:|---:|---:|---:|---:|
| **ALL events (the verdict)** | 2,625 | +6.80% | +9.11% | **−2.31%** | **−1.85** | 68.3% |
| ladder deployed (≥ 2 fills) | 2,035 | +4.45% | −1.21% | +5.66% | 14.43 | 86.5% |
| never fell 7.5% (1 fill) | 590 | +14.90% | +44.71% | −29.81% | −12.55 | 5.3% |

The conditional split both priors predicted happened, and the sides do not balance. When the
ladder deploys (77.5% of events — these entries fire at five-year PE lows, and 58.6% of them
fell ANOTHER 15%), the pyramid wins big (+5.66%, t = 14.43). But in the 22.5% of events where
the stock never gave back 7.5%, two thirds of the account sat in cash through a +44.7% average
run (−29.81%). The pyramid wins 68.3% of all events and still loses the mean — a
high-hit-rate, negative-skew execution rule. Risk-adjusted it is a dead tie (cross-sectional
mean/sd 0.178 vs 0.179), so the ladder does not even buy smoothness, just a different failure
mode. Verdict needed pyramid > lump with t > 2; got −2.31%, t = **−1.85**. **REJECTED.**

## #4 Inverse-PE on cyclicals — REJECTED (timing subtracts, in both directions)

`python scripts/cyclical_pe_study.py` — 434 cyclical names (tech + infrastructure buckets),
~203 held by buy-and-hold vs ~58 by the rule on an average day.

| variant | wealth | bench | active/yr | t(month) |
|---|---:|---:|---:|---:|
| **P_inv vs hold-all NET (the verdict)** | 1.983 | 2.114 | **−1.79%** | **−0.28** |
| P_mirror (buy low PE) vs hold-all | 1.905 | 2.114 | −2.73% | −0.43 |
| P_inv vs P_mirror | 1.983 | 1.905 | +0.94% | 0.25 |
| tech bucket only | 2.326 | 2.229 | −0.91% | −0.13 |
| infrastructure only | 0.779 | 1.188 | −4.88% | −0.81 |

The friend's inversion does beat its own mirror image (+0.94%/yr, t 0.25 — noise), so the
DIRECTION intuition is not refuted; but both PE-timing directions lose to simply holding the
names, and the infrastructure sleeve — the textbook cyclical — is where the rule does worst.
The one rule the friend flagged as "the part I am least sure about" is the right one to doubt:
on this investable universe, peTTM percentiles time nothing. Verdict needed P_inv > hold-all
with monthly t > 2 AND P_inv > P_mirror; got t = −0.28. **REJECTED.**

## #5 The 120-day box — REJECTED (the stamp tax is innocent; the signal is backwards)

`python scripts/box_trading_study.py` — 106 non-cyclical quality names (primary), all PIT
HS300 names (robustness).

| variant | excess/yr | Sharpe(diff) | t(month) |
|---|---:|---:|---:|
| **primary NET (the verdict)** | **−2.57%** | −0.59 | **−2.23** |
| primary GROSS (zero frictions) | −2.49% | −0.57 | −2.16 |
| all PIT HS300 names | −1.66% | −0.37 | −1.53 |

The pre-registered killer hypothesis — 做T frequency times stamp tax eats the edge — is wrong
in the most instructive way: total friction drag is 0.08%/yr (the box trades only ~7.9×
account value in 11.5 years), and the strategy loses −2.49%/yr GROSS. The signal itself is
backwards on these names: a quality name at the top of its 120-day box is more often breaking
out than reverting, so the rule sells half into strength again and again and holds full size
through the fades. 24.5% of names beat plain holding; both eras negative. This is the EOD
operationalization disclosed in the issue — a skilled intraday trader may do better, but the
mechanical box as stated is significantly NEGATIVE (t = −2.23). **REJECTED.**

## #6 Margin-balance timing — REJECTED (it buys safety, not wealth)

`python scripts/margin_timing_study.py` — 2,800 balance days, 2015-01 → present; the frozen
machine produced just **5 round trips** and sat in the market only **26.6%** of days.

| variant | wealth | b&h | Sharpe | b&h | t(month) | in-mkt |
|---|---:|---:|---:|---:|---:|---:|
| **frozen rule NET (the verdict)** | 1.636 | 1.684 | 0.48 | 0.32 | **−0.50** | 26.6% |
| entry leg only | 1.508 | 1.684 | 0.29 | 0.32 | −0.53 | 89.9% |
| exit leg only | 2.141 | 1.684 | 0.44 | 0.32 | 0.57 | 62.6% |
| 2015–2019 | 1.439 | 1.118 | 0.81 | 0.23 | 0.16 | 22.1% |
| 2020–2026 | 1.137 | 1.507 | 0.25 | 0.43 | −1.11 | 30.1% |

The rule's one great call is real: it sold before the 2015 crash (2015–2019 wealth 1.439 vs
1.118 holding). Then it spent the next six years mostly in cash while the market recovered
(2020–2026: 1.137 vs 1.507). Sharpe improves (0.48 vs 0.32) because cash is calm, but the
frozen verdict required wealth AND Sharpe with t > 2 — final wealth is BELOW buy-and-hold and
t = −0.50. All four ±10-percentile-point threshold variants also end below buy-and-hold
(wealth 1.411–1.613), so this is not a knife-edge parameterization. The attribution is
one-sided: the exit leg alone (sell above the 80th percentile, otherwise stay in) ends at
2.141 vs 1.684 — the only leg with any wealth edge, though at t = 0.57 it is indistinguishable
from luck. The entry leg (wait for the trough reversal) only costs. **REJECTED.**

## #7 The Premier's symposium — REJECTED (attendance carries no information)

`python scripts/symposium_study.py` — the first crack at the selection layer: 24 symposiums
(2015-04 → 2026-07) compiled AFTER the issue froze the source hierarchy and mapping rules,
85 named entrepreneur speeches, 43 mapped to A-share tickers (list with one official source
URL per event: `data/manual/symposium_events.csv`; flagship subsidiaries resolved by total
market cap at the event date, computations recorded in the CSV notes).

| horizon (net, vs HS300 EW) | N | mean | median | t(month) | hit |
|---|---:|---:|---:|---:|---:|
| 20d | 40 | +0.36% | −3.11% | 0.65 | 32.6% |
| 60d | 40 | +2.58% | −3.50% | 0.59 | 41.9% |
| **250d (the verdict)** | 38 | **+0.04%** | −1.55% | **0.08** | 41.9% |

As close to zero as an event study gets. The shape is a lottery: the median invitee
UNDERPERFORMS the size-matched benchmark at every horizon while a few big winners pull the
mean back to flat; against the full-universe EW benchmark the events show −2.31% (the size
confound the frozen HS300 benchmark exists to remove). Both premiers' eras are flat;
Li-Qiang-era invitees skew weak (median −6.26%). Attendance as a buy signal is dead; the
friend's symposium idea survives only as an unregistered CONTENT claim (which sectors get
discussed), explicitly not tested here. **REJECTED.**

## Synthesis — five rules, five REJECTED, one coherent picture

All five mechanical rules underperform their do-nothing benchmarks; none is significant in the
claimed direction, and two (#3's no-drawdown branch, #5) are significantly negative. The
failures are not five separate accidents — they are one theme seen from five angles: **every
rule in the system sells strength or waits for weakness** (buy only at PE lows, scale in only
on further falls, lighten at the box top, stand aside until margin balance troughs), and on
this universe in this window, strength kept going and weakness kept going. Mean-reversion
instincts priced for a range-bound market met a market that trends.

Three nuances the numbers force on that summary, in fairness to the system's author:

1. **The quality gate itself is fine** (+0.86%/yr, noise) — what fails is every timing layer
   stacked on top of it. "Buy good companies" survives; "buy them cheap by their own history,
   add falling, trim rising, time the mood" each subtract.
2. **The risk-management story is real but does not pay.** The pyramid wins 68% of events;
   margin timing improves Sharpe and dodged 2015. If the objective were "smallest chance of
   feeling bad," several rules deliver. The frozen verdicts asked whether they make MONEY
   versus doing nothing, and none does.
3. **The friend's own doubt was well-calibrated**: the single rule they flagged as least
   certain (cyclical inverse-PE) is genuinely directionless (beats its mirror, loses to
   holding) — while the rule they were most confident in (#2) was the largest detractor.

The discretionary selection layer got one mechanical probe (#7: symposium attendance — no
information) and otherwise stays out of scope. A point-in-time forward registration of the
friend's actual watchlist opened on 2026-07-15 (hash commitment in issue #2's thread;
plaintext off-repo) — if their real edge lives in selection rather than rules, that record is
what will eventually show it.

## #8 A second friend: the conviction-sector creed — CONFIRMED, and financially empty

`python scripts/sector_creed_study.py` — a different friend's system (relayed with consent):
pick your sector, add on every dip, hold — "it always comes back." The creed was frozen as:
the −20/−30/−40% ladder recovers its averaged-down cost within 500 trading days, in ≥ 90% of
≥ 20%-drawdown episodes across all 31 Shenwan level-1 indices, with a deliberately generous
+2%/yr dividend credit. Data: `data/parquet/sw_indices.parquet` (2000 → present); 202
episodes, 173 decided.

**Result: 99.4% recover — CONFIRMED as frozen — and the confirmation is the finding.** The
median episode touches break-even ON DAY ONE (the ladder is only a third invested and
cost-anchored, so any bounce clears the bar); the friend's own sectors run 23/23. But the
median episode ends the 500 days at **0.978x**, **56.6% finish below cost after having
touched it**, the worst episodes finish near 0.55x (agriculture 2001, media/computers 2015),
and the identical tranche schedule wired into the HS300 instead of the conviction sector ends
at the same place (t = 1.53). The creed is true the way a fishing float bobs: it surfaces, it
does not stay up — cost-anchoring plus partial investment make "it came back" almost
unfalsifiable, which is precisely why it feels so reliable from inside. The one genuine
failure (agriculture, triggered 2001-09) took the better part of five years to surface — the
regime-death tail the creed has no answer for.

## #9 The dividend-yield band on banks — REJECTED (positive but unproven; the example name is the counterexample)

`python scripts/dividend_band_study.py` — the first friend's own if-then ("buy above 5.5%
yield, sell below 4.5%"), tested on their high-dividend sleeve: 41 lake banks, 490
ex-dividend events, point-in-time trailing yield over the unadjusted close.

| variant | wealth | bench | active/yr | t(month) |
|---|---:|---:|---:|---:|
| **P_band vs all-banks EW NET (the verdict)** | 2.376 | 1.531 | **+3.31%** | **1.09** |
| 2020–2026 | 1.672 | 1.282 | +3.88% | 2.23 |
| all four ±0.5pp threshold variants | 2.16–2.47 | 1.531 | +2.2% to +3.9% | 0.59–1.48 |
| the single-name example | 1.744 | 4.039 | −10.90% | −1.48 |

The strongest point estimate of the series — +3.3%/yr over owning all banks, better Sharpe,
every threshold variant positive — and still REJECTED: the full-window monthly-clustered t
is 1.09 against the frozen bar of 2. Eleven years of active return that arrives in a few
repricing waves cannot prove itself at the pre-registered standard; recorded as "positive
but unproven," not "worthless." The telling detail: on the very bank the friend quoted, the
band exits whenever the yield compresses — i.e., whenever the stock re-rates — and finished
at 1.74x against 4.04x for holding it. A yield band is value timing in income clothing: it
trades away from the sector's strongest name and lives off the laggards' mean reversion.
