# The dividend-reinvestment accounting gap

Pre-registered as [issue #24](https://github.com/Rhymer-Lcy/hermes-quant/issues/24) before any
confirmatory number existed. Module `src/hermes/research/backtest/dividends.py`; script
`scripts/dividend_tax_study.py`; tests `tests/test_dividends.py`.

**Verdict: the backtest overstates the deployed strategy by about +0.30 pp/yr, or roughly 3% of
its net return.** The strategy is unchanged and no parameter moved. What changes is the number
hermes is allowed to quote about itself.

## The question, and why it is not the one that was asked

An outside investor asked whether hermes accounts for dividend reinvestment -- the share-count
growth and cost-basis reduction a holder gets when dividends are put back to work.

It does, and always has. The lake is pulled forward-adjusted (BaoStock `adjustflag="2"`), and that
series is a **total-return** series: on an ex-date the adjusted return already contains
`dps / raw_close[t-1]`. So the engine reinvests every dividend at the ex-date close, instantly,
pre-tax, in fractional shares, into the same name. All four of those are better than any real
individual account can do. **The error therefore runs one way: the reported curve is an
overstatement, not an omission.**

### The calibration that established this (a data property, measured before the design)

Over 1,447 pure-cash ex-date events in a 200-name random sample (seed 0), 2015-03-26 to 2026-07-16:

| statistic | value |
|---|---|
| median `adj_ret - tr_ret` | -2.4e-05 |
| p95 abs `adj_ret - tr_ret` | 1.4e-03 |
| share within 5e-4 of the **total-return** identity | 77.4% |
| share within 5e-4 of the **price-only** return | 3.3% |

Worked example, `sh.600808` ex 2022-07-21, DPS 0.35 on a prior close of 3.69 (a 9.5% yield):
raw return **-9.76%**, adjusted return **-0.30%**, total return **-0.27%**.

The residual is price quantisation: raw closes are quoted to two decimals, so a CNY 2 stock carries
~0.25% of rounding in a single-bar return. It does not propagate -- the measurement below works
entirely in adjusted space and never divides by a rounded price difference.

## Method: perturb the panel, not the engine

Because the gross dividend enters the adjusted return as exactly `dps / raw_close[t-1]`,
withholding a rate `tau` is a closed-form perturbation and needs **no engine change at all**:

```
net_ret[t]   = adj_ret[t] - tau * dps[t] / raw_close[t-1]
factor[t]    = (1 + net_ret[t]) / (1 + adj_ret[t]) = 1 - tau * dyield / (1 + adj_ret)
net_close[t] = adj_close[t] * cumprod(factor)[t]
```

At `tau == 0` every factor is exactly `1.0`, so the panel is bit-identical to canonical **by
construction** rather than by tolerance. That is the direct analogue of the D=1 parity gate in
[#21](rebalance_timing_study.md): if the neutral setting is not byte-identical, no other setting
can be trusted.

`tau` is not a constant. It is the share-weighted statutory rate on the lots the book was actually
holding at each ex-date, matched FIFO from the engine's own fill log.

### The tax schedule

Caishui [2015] No. 101, individual investors. The holding period runs from purchase to **transfer**:
CSDCC withholds nothing at the ex-date for holdings of a year or less and settles the liability when
the shares are sold.

| holding period | inclusion | effective rate |
|---|---|---|
| `<= 1 month` | 100% | 20% |
| `1 month` to `1 year` | 50% | 10% |
| `> 1 year` | exempt | 0% |

Entitlement follows the register, not the calendar: the register closes at the end of the prior
trading day and the engine trades at the close, so a lot is on it iff `buy_date < ex <= sell_date`.
A lot bought at the ex-date close is too late; a lot sold at the prior close is already gone.

## Gates

| gate | result |
|---|---|
| **M** -- the derived share multiplier | **PASS** |
| **C** -- lake completeness over the held names | **PASS** |
| **P** -- bit-identical parity at `tau = 0` | **PASS**, all seven tiers |
| **A** -- engine rerun vs closed form, 5 bp | **FAIL as written** -- see below |

**Gate M.** Stock dividends are absent from the cash lake and would otherwise read as price
collapses, so the share multiplier is derived rather than assumed:
`m = ((1 + adj_ret) * raw_prev - dps) / raw_now`. Across 6,512 events the median `m` is
**1.000000**, 87.7% sit within `1 +/- 0.002`, and of the 616 events with `m > 1.05`, **95.0%** land
on a recognisable stock-dividend ratio -- 1.40 (119), 1.30 (101), 2.00 (97), 1.20 (86), 1.50 (69),
1.80 (23), 2.50 (19), 1.60 (18). A derivation that reproduces 10-for-4 and 10-for-10 to two
decimals is not fitting noise.

**Gate C.** All 77 names the book actually holds are present in `raw_close.parquet` and complete in
the dividend pull. (The A6 lesson: a study run against a silently partial lake is how a verdict
inverts.)

**Gate P.** In all seven tiers the `tau = 0` panel is bit-identical to canonical and so is the
resulting equity curve -- `DataFrame.equals`, not a tolerance.

### Gate A failed, and the failure is reported rather than accommodated

The pre-registration required the engine rerun and the closed form to agree within **5 bp of
terminal wealth**, and said in advance that disagreement is "a defect, not a tolerance to be
widened". It failed:

| tier | max gap | | tier | max gap |
|---|---|---|---|---|
| 10,000 | 641.0 bp FAIL | | 500,000 | 28.6 bp FAIL |
| 30,000 | 325.0 bp FAIL | | 1,000,000 | 8.4 bp FAIL |
| 50,000 | 150.0 bp FAIL | | 5,000,000 | **3.4 bp PASS** |
| 100,000 | 232.1 bp FAIL | | | |

**The 5 bp limit is left exactly as registered and the failure stands on record.** What follows is
diagnosis, not repair.

The gap collapses monotonically as the account grows, and reaches the frozen limit at the largest
tier. That is the signature of lot quantisation, not of a sign or scale error -- and it was
confirmed by a controlled experiment rather than inferred. Holding everything else fixed and
changing only the lot size from 100 shares to 1:

| tier | gap at lot 100 | gap at lot 1 |
|---|---|---|
| 30,000 | 325.0 bp | 70.4 bp |
| 100,000 | 232.1 bp | 16.7 bp |
| 10,000 | 555.2 bp | 431.1 bp |

The CNY 10k tier does not respond, because it is infeasible for a different reason already
documented in `live/strategy.py`: the CNY 5 minimum commission, not the lot, is what binds there.

**A post-hoc diagnostic, not part of the frozen design**, then measured the engine-rerun method's
own resolution: a sign-randomised placebo applying the same corrections, at the same dates, with
the same magnitudes, but with a coin flip deciding debit or credit. Its expected drag is zero, so
its spread is the noise floor.

| tier | placebo mean | placebo sd | 90% band (pp/yr) |
|---|---|---|---|
| 10,000 | +0.419 | 0.262 | [+0.018, +0.650] |
| 30,000 | -0.038 | 0.044 | [-0.109, +0.028] |
| 50,000 | +0.014 | 0.026 | [-0.030, +0.055] |
| 100,000 | +0.012 | 0.033 | [-0.038, +0.049] |
| 500,000 | +0.026 | 0.036 | [-0.020, +0.072] |
| 1,000,000 | +0.020 | 0.037 | [-0.026, +0.068] |
| 5,000,000 | +0.018 | 0.039 | [-0.031, +0.065] |

At CNY 10k a placebo that should cost nothing costs **+0.42 pp/yr on average with a 0.26 pp sd** --
wider than the effect being measured. That tier's engine-rerun number carries no information about
dividends, and the positive mean is itself a finding: at a capital-constrained tier any disturbance
to lot affordability is asymmetrically costly, in either direction.

The closed form is immune -- it perturbs the canonical curve arithmetically and never re-rounds a
lot -- so it is the estimator quoted below, with the engine rerun reported beside it.

## Results

### Drag in pp/yr, closed form, every frozen cell

| tier | canonical CAGR | tau-A | tau-B | tau-C | tau-D | tau-E | tau-F | tau-G |
|---|---|---|---|---|---|---|---|---|
| 10,000 | 6.56% | +0.252 | +0.854 | +0.428 | 0.000 | +0.243 | +0.396 | 0.000 |
| 30,000 | 8.47% | +0.280 | +0.996 | +0.499 | 0.000 | +0.269 | +0.475 | 0.000 |
| 50,000 | 8.77% | +0.282 | +1.020 | +0.511 | 0.000 | +0.271 | +0.486 | 0.000 |
| 100,000 | 9.05% | +0.289 | +1.042 | +0.522 | 0.000 | +0.278 | +0.497 | 0.000 |
| 500,000 | 9.42% | +0.295 | +1.060 | +0.531 | 0.000 | +0.284 | +0.507 | 0.000 |
| 1,000,000 | 9.44% | +0.295 | +1.062 | +0.532 | 0.000 | +0.284 | +0.508 | 0.000 |
| 5,000,000 | 9.47% | **+0.296** | +1.064 | +0.533 | 0.000 | +0.285 | +0.509 | 0.000 |

Engine rerun, same cells, for comparison: +0.863 / +0.183 / +0.312 / +0.187 / +0.312 / +0.302 /
**+0.298** for tau-A. At the two largest tiers -- the only ones with resolution -- the two methods
agree to within 0.01 pp/yr.

### The headline

**+0.30 pp/yr** on the deployed strategy, statutory rates, individual investor. Against a canonical
9.05-9.47% net CAGR that is **3.1-3.3% of the return**, and it is one-signed: it can only subtract.

The effect is stable rather than concentrated. Tax paid as a percentage of average equity, by
calendar year, at the CNY 100k tier:

| 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.24 | 0.08 | 0.07 | 0.17 | 0.40 | 0.33 | 0.48 | 0.34 | 0.32 | 0.33 | 0.17 | 0.12 |

mean 0.256 pp, sd 0.133 pp, min 0.070 (2017), max 0.481 (2021). No single year carries it.

### Why the blended rate is 5.6% and not 15%

The effective rate is **5.56%**, far below the 10-20% the pre-registration predicted. The reason is
visible in the band split, weighted by dividend **cash** (CNY 1M tier):

| band | share of dividend value |
|---|---|
| `<= 1 month` at 20% | 5.3% |
| `1 month` to `1 year` at 10% | 45.0% |
| `> 1 year`, exempt | **49.7%** |

A monthly rebalance does not imply monthly holdings. The deployed book is a low-PE value sleeve, and
its lots run 22 days minimum, 185 days median, 2,557 days maximum; 32.6% of lots are held past a
year. Those long lots span several ex-dates each while a one-month lot usually spans none, so
half of all dividend value is collected inside the exempt band even though only a third of lots
reach it.

The monthly gap also lands *just past* the one-month line rather than inside it, which is why the
20% band holds only 5.3%: consecutive month-end rebalances are typically 30-31 days apart, and one
calendar month after a purchase usually falls a day or two short of the next execution.

### Magnitude reconciliation, and a prediction that was right for the wrong reason

The book's realised gross dividend yield is **4.72%/yr** (CNY 5M tier; 3.86% at CNY 10k, where cash
drag dilutes it). That is well above HS300's 2-3%, and it should be: the holdings are a deep-value
low-PE basket -- 光大银行, 中国银行, 交通银行, 兴业银行, 民生银行, 农业银行, 浦发银行, 中国建筑,
中国中冶 -- i.e. banks and construction.

An **independent** cross-check through a different code path (`trailing_yield` over unadjusted
closes, value-weighted by the actual holdings) gives a mean trailing yield of **5.13%** against the
study's realised **4.71%**. The two share no code. The residual is expected in sign: trailing yield
looks back a year while realised yield is the cash actually received forward, and A-share dividends
were falling into 2026.

**This is worth recording against myself.** The pre-registration predicted a drag of 0.2-0.8 pp/yr
from a portfolio yield of 2-4% and a blended rate of 10-20%. The measured drag, +0.30 pp/yr, is
inside that band -- but **both inputs were wrong, in opposite directions**: the yield was 4.7% (too
low a guess) and the rate 5.6% (too high a guess), and the two errors cancelled in the product. The
band was hit by luck, not by understanding. A prediction interval that is right for compensating
reasons is not evidence the mechanism was understood, and had only one of the two errors occurred
the prediction would have missed by a factor of two.

### Second-order components, reported separately

Neither is folded into the headline.

*Cash-timing.* In the model a dividend compounds inside the name from the ex-date; in reality it
sits as idle cash until the next rebalance. Measured exactly rather than bounded, the effect is
**-0.008 to +0.008 pp/yr** -- and at every tier above CNY 10k it is *negative*, i.e. holding the
cash idle marginally helped, because the names drifted down over those gaps on average. It is
noise at this scale either way.

*Lot and minimum-commission friction on reinvestment.* Dividend cash is not stranded: it merges with
the rebalance cash and is spent on the same 100-share lots the engine already models, so its only
cost is the idle period above. Per-rebalance dividend cash is CNY 50 at the 10k tier and CNY 36,018
at 5M -- at the small end far below one lot, which is why it cannot be reinvested on its own and why
the timing term is all there is.

### A convention the pre-registration did not freeze

A lot still open at the window end has not been transferred, so the statute has not yet fixed its
rate. The study continues it to the window edge. **That was not frozen in advance**, so its range
is reported rather than asserted immaterial:

| tier | open lots | registered | all-exempt | all-20% (bound) |
|---|---|---|---|---|
| 100,000 | 95 / 975 | +0.289 | +0.280 | +0.391 |
| 1,000,000 | 125 / 1,333 | +0.295 | +0.287 | +0.401 |
| 5,000,000 | 129 / 1,365 | +0.296 | +0.287 | +0.402 |

The realistic range is registered-to-exempt, a spread of **0.009 pp/yr** -- immaterial. The all-20%
column is a strict upper bound and *not a scenario*: under the statute a lot cannot both collect a
dividend months after purchase and be taxed at the one-month rate. Its extra tax also lands entirely
at the end of the window, where it has the least time to compound, which is why the drag rises by
36% while the tax rises by 51%.

## What this changes

1. **`docs/paper_trading.md` no longer says "second-order".** It says +0.30 pp/yr.
2. **Any outward statement of hermes's return carries this haircut**, or states that it does not.
   The 9.05% at the CNY 100k tier is a pre-tax-on-dividends figure; an individual investor's
   comparable expectation is ~8.76%.
3. **It does not apply to everyone.** tau-G is the institutional cell: a corporate holder is exempt
   and the drag is zero. The number is a property of the *holder*, not of the strategy.
4. **Nothing about the strategy changes.** `DEPLOYED`, `PAPER_INCEPTION`, `results/paper/` and every
   engine default path are untouched, and the `tau = 0` cell reproduces canonical bit-for-bit.

## The temptation this study refuses

The one-month tax boundary sits almost exactly where a monthly rebalance lands, and the results show
the 20% band holds only 5.3% of dividend value precisely because executions usually fall a day or
two *past* it. It is therefore obvious how to shave the drag further by nudging the rebalance date.

**That change is not made here, and the pre-registration ruled it out before the numbers existed.**
It would be a strategy modification chosen with knowledge of the outcome -- exactly the 31-choose-1
selection problem [#21](rebalance_timing_study.md) was built to avoid, and one whose plausible prize
(a fraction of 0.30 pp/yr) is far smaller than the noise floor measured above. If it is ever
pursued it gets its own pre-registration and its own forward record.

## Reproducing

```
conda activate hermes
python scripts/dividend_tax_study.py        # ~3 min, writes results/backtests/dividend_tax/
pytest tests/test_dividends.py -q           # 32 tests
```

Two of those tests are regressions for defects this study's own review found: the band split was
first weighted by share count rather than by cash (a cheap name and an expensive one counted equally
per share, moving the split by ~5 pp), and the open-lot sensitivity first forced open lots into the
20% band by collapsing `sell_date` onto `buy_date`, which silently revoked their register
entitlement instead of re-rating them. Both were caught by an independent brute-force recomputation
that agrees with the module to **0.0000 bp**, not by re-reading the code.
