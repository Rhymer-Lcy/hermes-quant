# Drawdown control (turning a signal into something deployable)

Plain value (earnings yield), survivorship-free PIT, top-10 equal weight: **+9.2% CAGR
but -33% max drawdown** -- not deployable as-is. This log records attempts to cut the
drawdown without giving back the return. The bar: raise **Calmar = CAGR / |maxDD|**.

## A1 — market-regime filter (沪深300 200-day MA): FAILED

Gate gross exposure to 0 when 沪深300 is below its 200-day MA, else 1 (decided monthly,
at the rebalance; PIT universe; A-share frictions).

| variant                    | CAGR  | maxDD  | Calmar |
|----------------------------|------:|-------:|-------:|
| value                      | +9.2% | -33.0% | 0.28   |
| value + 200d-MA regime     | +0.7% | -42.5% | 0.02   |

The filter hurt both axes: it sharply reduced the return (9.2% -> 0.7%) and even worsened max
drawdown (-33% -> -42.5%). Why: A-share moves are sharp and V-shaped, so a lagging
200-day MA exits *after* the drop and re-enters *after* the recovery (whipsaw); the
monthly rebalance compounds the lag; and value's worst stretches are growth-led rallies
where the index is *above* its MA (risk-on) while cheap names keep falling -- exactly
when the filter stays fully invested. Trend-timing the index is not the tool here.

Reframe: a -33% drawdown is not extreme for long-only A-share equity (the index itself
saw large drawdowns over 2015-2025). The next lever is **position-level risk**
(volatility targeting, per-name and sector weight caps), not market timing. -> A2.

(The `exposure_asof` control added to the backtest is retained and reusable; it is the
filter *policy* -- a binary MA gate -- that failed, not the mechanism.)

## A2 — position-level risk (inverse-vol weighting + value×low-vol blend): PARTIAL

Two independent levers, holding everything else fixed (PIT HS300, A-share frictions,
top-10 monthly), evaluated as a 2×2. Inverse-vol redistributes *within* the basket at
the **same gross exposure** as equal weight (so the comparison isolates weighting); the
low-vol blend changes *which* names are selected. Bar: raise Calmar over plain value.

| variant                | CAGR (100k / 1M) | maxDD (100k / 1M) | Calmar |
|------------------------|-----------------:|------------------:|-------:|
| value / equal (base)   | +9.2% / +9.5%    | -33.0% / -33.6%   | 0.28   |
| value / **inverse-vol**| +9.8% / +10.1%   | -32.8% / -33.1%   | **0.30** |
| val×lowvol / equal     | +9.0% / +9.3%    | -32.4% / -32.9%   | 0.28   |
| val×lowvol / inverse-vol | +9.2% / +9.7%  | -32.2% / -32.7%   | 0.30   |

**Inverse-vol weighting is the only lever that survives.** A small, *robust* Calmar lift
(0.28 → 0.30 at both tiers, stable across lookbacks 20–120d), almost entirely from CAGR
(~+0.5pp) with drawdown ~unchanged -- grounded in the low-vol anomaly (calmer names earn
better risk-adjusted returns). Worth keeping. The **value×low-vol blend adds nothing**:
val×lowvol/invvol equals value/invvol, i.e. the blend contributes 0 beyond inverse-vol.

### Key finding: the survivorship leak this caught

The blend *first* looked like a clear win (Calmar 0.32, with a clean monotone window
sweep 0.35→0.26). Adversarial verification found why: the blend standardizes
cross-sectionally, and it was standardizing over the survivorship-defined **union** (657
names *ever* in HS300), not the then-current members. That leaks future membership into
the z-scores -- a member's standardized score depended on names present in the panel only
because they *join* the index later. Restricting standardization to the PIT members
(`factors.library.restrict_to_universe`) collapsed the blend's Calmar **0.32 → 0.28** and
turned the "clean monotone" window sweep into noise (0.31/0.36/0.28/0.27/0.28). The
apparent edge was the leak. Single-factor value and inverse-vol were never affected (they
do no cross-sectional standardization). Guardrail added: `restrict_to_universe` plus
hard docstring warnings on `winsorize_xs`/`zscore_xs`/`standardize`/`blend`.

### Conclusion

The −33% drawdown is **systematic** -- ten correlated HS300 blue-chips fall together, so
intra-basket levers barely dent it (A2's best honest maxDD is −32.7%, vs −33.6%). This is
the same verdict as A1 from the other side: the drawdown is a whole-market phenomenon, not
something selection/weighting among large caps can fix. Inverse-vol weighting is adopted as
a small, clean, free improvement; the low-vol blend is shelved. Cutting the systematic
drawdown materially would need either genuine breadth (uncorrelated factors / more names /
smaller caps) or a non-naive hedge -- not a position-level tweak. → next is **breadth**
(factor diversification, option B), not more drawdown plumbing.

**Update — option B done (see [docs/multi_factor.md](multi_factor.md)):** factor
diversification *did* raise Calmar (a light 1-month-reversal tilt, 0.28 → 0.32, above A2's
0.30, with gross/zero-cost Calmar rising too, so it is real alpha) -- but **maxDD stayed
~−33%**. Breadth helped the numerator (return), not the drawdown, confirming yet again that
−33% is systematic. The two remaining *selection-side* levers were then tested and both
FAILED (A3, A4 below); only the *universe* dimension (CSI 500) is left untried.

## A3 — sector / industry neutralization: FAILED (counterproductive)

Premise (correct): the value top-10 is extremely sector-concentrated -- ~70% banks (申万
J-financials) plus real-estate and construction. So "diversify across sectors to cut the
correlated drawdown" looks reasonable. Tested two ways on PIT HS300 (top-10 monthly, A-share
frictions, 1M): full within-sector demean of the value score, and a per-sector name cap.

| variant                     | CAGR  | maxDD  | Calmar |
|-----------------------------|------:|-------:|-------:|
| value (base)                | +9.5% | -33.6% | 0.28   |
| value, full sector-demean   | +0.3% | -58.1% | 0.01   |
| value, per-sector cap 5     | +8.1% | -39.4% | 0.21   |
| value, per-sector cap 4     | +7.8% | -41.9% | 0.19   |
| value, per-sector cap 3     | +7.5% | -45.1% | 0.17   |
| value, per-sector cap 2     | +6.2% | -50.9% | 0.12   |

**Every degree of sector diversification makes the drawdown WORSE, monotonically** (the full
curve was swept -- IRON RULE 2). Why: A-share banks trade ~4× cheaper on earnings yield than
everything else *and* are the lowest-vol, most defensive cluster in HS300, so the value
factor essentially *is* the bank trade, and that concentration is the **source of the
strategy's residual defensiveness**, not the source of the −33%. Forcing money out of banks
injects higher-beta non-financial names that fall harder in a whole-market selloff. Note this
test used the *latest* Shenwan snapshot on all dates -- a mild look-ahead *in neutralization's
favour* -- and it still fails, so the null is robust. (`baostock_source.stock_industry` is
free and PIT-capable; kept for sector *attribution*, not as an alpha lever. Repro:
`scripts/a3_sector_demo.py`.)

## A4 — size tilt (small-cap): FAILED (counterproductive)

Free-float cap reconstructs from the existing daily lake with no Tushare pull --
`float_cap = amount / (turn/100)` (BaoStock `turn` is the free-float turnover rate), 97.7%
bar coverage, validated against known mega-caps (600519 ≈ 1.73万亿). So the old "size is
blocked on a rate-limited Tushare tier" caveat is moot; the factor is now testable. Swept
value:size at PIT HS300 (top-10 monthly, frictions, 1M; full curve -- IRON RULE 2):

| variant        | CAGR  | maxDD  | net Calmar | gross Calmar |
|----------------|------:|-------:|-----------:|-------------:|
| value (base)   | +9.5% | -33.6% | 0.28       | 0.30         |
| value + 0.2·size | +8.8% | -36.2% | 0.24     | 0.26         |
| value + 0.3·size | +8.7% | -34.8% | 0.25     | 0.27         |
| value + 0.5·size | +7.3% | -41.5% | 0.18     | 0.19         |
| size only      | -6.4% | -85.1% | -0.08      | -0.07        |

**Every size weight lowers CAGR and deepens maxDD; size-only delivers −85% maxDD (infeasible)**
(and worse at every capital tier). Within HS300 the universe is all large caps, so a "small" name
is a *demoted / falling-knife* blue-chip (distress beta), not the small-cap premium -- and it
co-moves with the market (corr(small, large) ≈ 0.76), adding no uncorrelated return. Same
verdict as A1/A2/A3/B. (Repro: `scripts/a4_size_demo.py`.)

## A6 — wider universe (CSI 500): FAILED (small-cap beta is worse, not better)

The last untested equity lever: extend to CSI 500 (中证500) mid/small caps for genuine
cross-sector breadth. Built survivorship-free (1326-name PIT union, free via BaoStock
`query_zz500_stocks`), traded with **涨跌停 no-fill ON and ST names filtered** (the rigorous
small-cap treatment), same 2015-2025 window as HS300 (`scripts/csi500_study.py`):

| variant (1M, 2015-2025)        | CAGR  | maxDD  | net Calmar | sector HHI |
|--------------------------------|------:|-------:|-----------:|-----------:|
| HS300 value+rev 5/1 (deployed) | +10.5%| -33.0% | **0.32**   | 0.58 |
| CSI500 value, top-30           | +4.6% | -46.1% | 0.10       | 0.13 |
| CSI500 value, top-50           | +5.1% | -47.8% | 0.11       | 0.09 |
| CSI500 value+quality+rev top30 | +0.0% | -69.7% | 0.00       | 0.11 |

**Breadth materialized -- and bought nothing.** The wider universe DOES cure the concentration
(sector HHI 0.58 → 0.09-0.13, the ~70%-banks problem gone), but maxDD is far WORSE (-46% to -48%
vs -33%) and CAGR far LOWER (+4-5% vs +10.5%), so net Calmar collapses 0.32 → 0.10. It is the
**universe, not the friction model**: CSI500 value top-30 at ZERO cost and limits OFF is still
-45.2% maxDD / 0.10 Calmar (limits add ~1%). A-share mid/small caps simply carry deeper systematic
drawdowns (2015/2018 crashes) and lower risk-adjusted returns; more names re-sample a *worse* beta.
Quality/diversification (which only cured concentration on HS300) make CSI500 worse still (-70%).
**Verdict: stay on HS300; CSI500 expansion is rejected.**

### Where this leaves the drawdown

**Six independent angles -- A1 (timing), A2 (weighting), B (factor breadth), A3 (sector), A4
(size), A6 (wider universe) -- plus the quality/multi-factor study, all confirm the same thing:
the deployed HS300 value + light-reversal book (Calmar ~0.32, maxDD −33%) is the best long-only
A-share equity strategy reachable by selection / weighting / universe.** The −33% is systematic
whole-market beta. **The ONE remaining lever that can actually cut it is a HEDGE OVERLAY**
(short 股指期货 IF / index puts) -- a new instrument scope, which is exactly what the
`hermes.intraday` futures line sets up (IF minute data is in hand). Until then the deployed
strategy accepts the −33% as understood, bounded, systematic market risk.
