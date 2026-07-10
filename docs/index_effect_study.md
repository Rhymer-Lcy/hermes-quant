# HS300 index-reconstitution effect: the US delete-reversal does NOT replicate here

The sibling plutus-quant validated one retail-operable satellite edge in US equities: S&P 500
DELETIONS earn a +3.4% to +5.1% abnormal bounce net of costs over 20-60 days. The mechanism
(index-tracker forced selling overshoots, then mean-reverts) is not US-specific, and hermes already
holds everything needed to test the A-share analogue: PIT month-end HS300 membership plus daily
bars for every name ever in the index. 450 deletions / 450 additions over 2015-2025 in 27
reconstitution batches — roughly twice the US sample. Reproduce: `python scripts/index_effect_study.py`.

What the design honestly measures: membership is month-END snapshots, so a change is detected
~2-3 weeks after its (mid-June / mid-December) effective date, and entry is the next trading day's
close — the version a retail replicator of this repo could actually trade, missing any front-loaded
part of the bounce. Abnormal return is vs the PIT-universe equal-weight mean; a halted day is no
trade (not a zero return); names that die inside the window stay in at their last price. Inference
is on batch means (27 independent batches), since same-batch events share their return window.

## Result (cumulative abnormal return from the lagged entry)

| event day | 5 | 10 | 20 | 40 | 60 |
|---|---:|---:|---:|---:|---:|
| DELETE mean CAAR | −0.13% | +0.00% | −0.17% | +1.00% | **+1.41%** |
| DELETE batch-t | −0.43 | +0.01 | −0.25 | +1.05 | +1.08 |
| ADD mean CAAR | −1.26% | −1.44% | −2.62% | −2.87% | **−3.36%** |
| ADD batch-t | −1.96 | −2.18 | −2.78 | −2.18 | **−3.09** |

## Verdict

- **DELETE-reversal: NOT established.** The 60-day read is positive (+1.41% gross, +1.20% net of
  the ~0.20% round-trip) but statistically indistinguishable from zero (t = +1.08, hit rate 48%
  — the mean rides on a minority of large bounces). At the granularity this repo can trade, the
  US satellite does not replicate. Two candidate explanations, not distinguishable from this data:
  the ~3-week detection lag misses a front-loaded bounce, or A-share deletions (typically demotions
  into CSI500, still heavily retail-held) simply lack the US-style forced-selling dislocation.
- **ADD-underperformance: significant, but unharvestable.** Fresh additions LAG the universe by
  ~3.4% over the following 60 days (t = −3.09, monotone across horizons) — consistent with the US
  finding that the inclusion run-up is over by the effective date and partly reverts. Harvesting it
  requires shorting, which retail A-share accounts cannot do; as a standalone rule it is only a
  "do not buy fresh additions" caution.

Per the pre-commitment in the script: no window/threshold tuning follows a weak read. The one
follow-up that could legitimately resolve the DELETE question is finer event anchoring — weekly
membership snapshots around the June/December reconstitution windows would let entry track the
actual effective date instead of lagging it by ~3 weeks. That is a data-collection task
(BaoStock membership queries are date-parameterized), not a re-analysis of this dataset, and it
is left explicitly unbuilt until wanted.
