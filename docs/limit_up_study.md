# After the limit-up seal there is nothing to buy — the bleed survives a different microstructure

Pre-registered as [issue #1](https://github.com/Rhymer-Lcy/hermes-quant/issues/1) — this repo's
first pre-registered study — frozen before any code existed. The question is the A-share twin of
the sibling repo's gap result ([plutus gap_lottery_study](https://github.com/Rhymer-Lcy/plutus-quant/blob/main/docs/gap_lottery_study.md):
any US stock that gaps up ≥ +20% overnight bleeds ~−3% abnormal over the next month). A-share
price limits FORBID that gap and smear the move across locked sessions, so two priors collided
head-on: lottery/attention (the entry bleeds) versus truncated price discovery (the 打板 thesis:
what remains is continuation, with academic support). Reproduce: `python scripts/limit_up_study.py`.

Universe: point-in-time HS300 + CSI500 members (survivorship-free monthly snapshots), non-ST,
≥ 20 prior traded days, 2015–present — **15,464 fresh limit-up-locked closes in 1,392 names**
(10 unresolved within 60 days). Event detection is date-aware (ChiNext ±10% before the
2020-08-24 reform, ±20% after; the detection bug that applied today's widths to the whole
history was fixed and unit-tested before the issue was filed). Entry: **the first later close
NOT limit-locked** — a close still sealed has no sellers, so the first unsealed close is the
first genuinely fillable EOD print. The run earned while sealed — mean **+2.35%**, max +125% —
is the part you cannot get, the A-share analog of the US overnight gap. Abnormal = the name
minus the same-day equal-weight PIT-member mean; costs 0.20% round trip; inference on
**monthly-clustered t** (2015 alone supplies 29% of events).

## Result: buy the first fillable close, and bleed

| horizon | gross | NET | median (net) | t(event) | **t(month)** | hit |
|---|---:|---:|---:|---:|---:|---:|
| 1d | −0.28% | −0.48% | −0.92% | −14.50 | −6.90 | 39.9% |
| 5d | −0.78% | −0.98% | −1.89% | −14.02 | −6.61 | 39.6% |
| 10d | −1.21% | −1.41% | −2.62% | −15.26 | −6.56 | 38.9% |
| **20d (the frozen verdict)** | −1.78% | **−1.98%** | −3.31% | −16.49 | **−5.88** | 39.3% |
| 60d | −2.62% | −2.82% | −5.06% | −15.86 | −5.58 | 38.4% |

## VERDICT: REJECTED — and the folklore's core claim inverts

The frozen rule (20d net positive, monthly t > 2) fails in the strongest possible way: the mean
is **negative with t = −5.88**. Robustness is crushing: winsorized 1% / 5% give t = −6.51 /
−8.37; the sign test rejects a coin flip at p = 1.5e−148 (39.6% positive); **all 12 calendar
years are negative**.

- **Seal strength predicts MORE bleed, not continuation.** Entry after a multi-day seal loses
  −2.93% vs −1.81% when fillable the next day. "强者恒强" — the heart of the 打板 thesis — runs
  backwards in this sample: the longer the queue held, the worse the first fill does.
- Both eras bleed: 2015–2019 −2.68% (t = −6.53), 2020–2026 −1.22% (t = −2.64). Thinner lately,
  still significant.
- HS300 members (−2.15%) and CSI500-only names (−1.91%) bleed alike; the 20%-width sample is
  small (291 events) and inconclusive, reported as such.
- **Cross-market synthesis:** the US gap bleeds −3%/20d, the A-share seal bleeds −2%/20d. The
  lottery/attention mechanism survives a completely different microstructure — the price limit
  does not convert reversal into continuation, it just stretches the execution out.

## The limit-down leg is a statistics lesson, not a trade

Buying the first fillable close after a fresh limit-DOWN prints an event-level t of **+8** at
20–60 days — and a monthly-clustered t that is **negative** (−1.85 to −2.98). The two statistics
point in opposite directions because limit-downs pile up in panic months and event-level
inference treats hundreds of same-month events as independent draws. Under the pre-registered
clustered statistic there is **no reliable knife-catching return**; the event-level "+8" is
exactly the artifact the design was frozen to avoid. Reported as context only — A-share retail
cannot short, so no verdict attaches.

## Limitations, kept from the pre-registration

- The 打板 folklore's home turf is micro-caps OUTSIDE these indices; this is the investable
  mid/large-cap version — the one a replicator of this repo could actually trade. The folklore
  may yet live in the untested micro-cap segment (a smaller, costlier, manipulation-prone pool).
- Entry is EOD close-to-close; intraday board-chasing (buying INTO the seal at the limit price
  and queueing) is a different trade this daily lake cannot price.
- ST names (±5% limits) are excluded rather than modeled; fresh listings' no-limit sessions are
  excluded by the 20-day history gate.
