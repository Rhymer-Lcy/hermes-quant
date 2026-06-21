# Drawdown control (turning a signal into something deployable)

Plain value (earnings yield), survivorship-free PIT, top-10 equal weight: +9.2% CAGR
but -33% max drawdown -- not deployable as-is. This log records attempts to cut the
drawdown without giving back the return. The bar is to raise **Calmar = CAGR / |maxDD|**.

## A1 — market-regime filter (CSI 300 200-day MA): FAILED

Gate gross exposure to 0 when CSI 300 is below its 200-day MA, else 1 (decided monthly,
at the rebalance; PIT universe; A-share frictions).

| variant                    | CAGR  | maxDD  | Calmar |
|----------------------------|------:|-------:|-------:|
| value                      | +9.2% | -33.0% | 0.28   |
| value + 200d-MA regime     | +0.7% | -42.5% | 0.02   |

The filter hurt both axes: it sharply reduced the return (9.2% -> 0.7%) and worsened max
drawdown (-33% -> -42.5%). Mechanism: A-share moves are sharp and V-shaped, so a lagging
200-day MA exits *after* the drop and re-enters *after* the recovery (whipsaw); the
monthly rebalance compounds the lag; and value's worst stretches are growth-led rallies
where the index is *above* its MA (risk-on) while cheap names keep falling -- precisely
when the filter stays fully invested. Trend-timing the index is not the appropriate tool here.

A -33% drawdown is not extreme for long-only A-share equity (the index itself
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

**Inverse-vol weighting is the only lever that survives.** A small but robust Calmar lift
(0.28 → 0.30 at both tiers, stable across lookbacks 20–120d), almost entirely from CAGR
(~+0.5pp) with drawdown approximately unchanged -- grounded in the low-vol anomaly (calmer names earn
better risk-adjusted returns). It is retained. The value×low-vol blend adds nothing:
val×lowvol/invvol equals value/invvol, i.e. the blend contributes nothing beyond inverse-vol.

### Key finding: the survivorship leak this caught

The blend initially looked like a clear win (Calmar 0.32, with a clean monotone window
sweep 0.35→0.26). Adversarial verification identified the cause: the blend standardizes
cross-sectionally, and it was standardizing over the survivorship-defined **union** (657
names *ever* in HS300), not the then-current members. That leaks future membership into
the z-scores -- a member's standardized score depended on names present in the panel only
because they *join* the index later. Restricting standardization to the PIT members
(`factors.library.restrict_to_universe`) collapsed the blend's Calmar **0.32 → 0.28** and
turned the "clean monotone" window sweep into noise (0.31/0.36/0.28/0.27/0.28). The
apparent edge was the leak. Single-factor value and inverse-vol were unaffected (they
do no cross-sectional standardization). Guardrail added: `restrict_to_universe` plus
docstring warnings on `winsorize_xs`/`zscore_xs`/`standardize`/`blend`.

### Conclusion

The −33% drawdown is **systematic** -- ten correlated HS300 blue-chips fall together, so
intra-basket levers barely dent it (A2's best maxDD is −32.7%, vs −33.6%). This is
the same verdict as A1 from the other side: the drawdown is a whole-market phenomenon, not
something selection/weighting among large caps can fix. Inverse-vol weighting is adopted as
a small, clean, no-cost improvement; the low-vol blend is shelved. Cutting the systematic
drawdown materially would require either genuine breadth (uncorrelated factors / more names /
smaller caps) or a non-naive hedge -- not a position-level tweak. The next lever is **breadth**
(factor diversification, option B).

**Update — option B done (see [docs/multi_factor.md](multi_factor.md)):** factor
diversification *did* raise Calmar (a light 1-month-reversal tilt, 0.28 → 0.32, above A2's
0.30, with gross/zero-cost Calmar rising too, so it is real alpha) -- but maxDD stayed
~−33%. Breadth helped the numerator (return), not the drawdown, again confirming that
−33% is systematic. The two remaining *selection-side* levers were then tested and both
FAILED (A3, A4 below); only the *universe* dimension (CSI 500) remained untried.

## A3 — sector / industry neutralization: FAILED (counterproductive)

Premise (correct): the value top-10 is heavily sector-concentrated -- ~70% banks (Shenwan
J-financials) plus real-estate and construction, so diversifying across sectors to cut the
correlated drawdown appears reasonable. Tested two ways on PIT HS300 (top-10 monthly, A-share
frictions, 1M): full within-sector demean of the value score, and a per-sector name cap.

| variant                     | CAGR  | maxDD  | Calmar |
|-----------------------------|------:|-------:|-------:|
| value (base)                | +9.5% | -33.6% | 0.28   |
| value, full sector-demean   | +0.3% | -58.1% | 0.01   |
| value, per-sector cap 5     | +8.1% | -39.4% | 0.21   |
| value, per-sector cap 4     | +7.8% | -41.9% | 0.19   |
| value, per-sector cap 3     | +7.5% | -45.1% | 0.17   |
| value, per-sector cap 2     | +6.2% | -50.9% | 0.12   |

**Every degree of sector diversification makes the drawdown monotonically worse** (the full
curve was swept before concluding). Mechanism: A-share banks trade ~4× cheaper on earnings yield than
everything else *and* are the lowest-vol, most defensive cluster in HS300, so the value
factor essentially *is* the bank trade, and that concentration is the **source of the
strategy's residual defensiveness**, not the source of the −33%. Forcing money out of banks
injects higher-beta non-financial names that fall harder in a whole-market selloff. This
test used the *latest* Shenwan snapshot on all dates -- a mild look-ahead *in neutralization's
favour* -- and it still fails, so the null is robust. (`baostock_source.stock_industry` is
free and PIT-capable; kept for sector *attribution*, not as an alpha lever. Repro:
`scripts/sector_neutral_study.py`.)

## A4 — size tilt (small-cap): FAILED (counterproductive)

Free-float cap reconstructs from the existing daily lake with no Tushare pull --
`float_cap = amount / (turn/100)` (BaoStock `turn` is the free-float turnover rate), 97.7%
bar coverage, validated against known mega-caps (600519 ≈ ¥1.73 trillion). The prior caveat that
size was blocked on a rate-limited Tushare tier no longer applies; the factor is now testable. Swept
value:size at PIT HS300 (top-10 monthly, frictions, 1M; full curve):

| variant        | CAGR  | maxDD  | net Calmar | gross Calmar |
|----------------|------:|-------:|-----------:|-------------:|
| value (base)   | +9.5% | -33.6% | 0.28       | 0.30         |
| value + 0.2·size | +8.8% | -36.2% | 0.24     | 0.26         |
| value + 0.3·size | +8.7% | -34.8% | 0.25     | 0.27         |
| value + 0.5·size | +7.3% | -41.5% | 0.18     | 0.19         |
| size only      | -6.4% | -85.1% | -0.08      | -0.07        |

**Every size weight lowers CAGR and deepens maxDD; size-only delivers −85% maxDD (infeasible)**
(and worse at every capital tier). Within HS300 the universe is all large caps, so a "small" name
is a *demoted, distressed* blue-chip (distress beta), not the small-cap premium -- and it
co-moves with the market (corr(small, large) ≈ 0.76), adding no uncorrelated return. Same
verdict as A1/A2/A3/B. (Repro: `scripts/size_tilt_study.py`.)

## A6 — wider universe (CSI 500): FAILED (small-cap beta is worse, not better)

The last untested equity lever: extend to CSI 500 mid/small caps for genuine
cross-sector breadth. Built survivorship-free (1326-name PIT union, free via BaoStock
`query_zz500_stocks`), traded with **price-limit no-fill ON and ST names filtered** (the rigorous
small-cap treatment), same 2015-2025 window as HS300 (`scripts/csi500_universe_study.py`):

| variant (1M, 2015-2025)        | CAGR  | maxDD  | net Calmar | sector HHI |
|--------------------------------|------:|-------:|-----------:|-----------:|
| HS300 value+rev 5/1 (deployed) | +10.5%| -33.0% | **0.32**   | 0.58 |
| CSI500 value, top-30           | +4.6% | -46.1% | 0.10       | 0.13 |
| CSI500 value, top-50           | +5.1% | -47.8% | 0.11       | 0.09 |
| CSI500 value+quality+rev top30 | +0.0% | -69.7% | 0.00       | 0.11 |

**Breadth materialized but delivered no improvement.** The wider universe does cure the concentration
(sector HHI 0.58 → 0.09-0.13, the ~70%-banks problem gone), but maxDD is far worse (-46% to -48%
vs -33%) and CAGR far lower (+4-5% vs +10.5%), so net Calmar collapses 0.32 → 0.10. It is the
**universe, not the friction model**: CSI500 value top-30 at zero cost and limits off is still
-45.2% maxDD / 0.10 Calmar (limits add ~1%). A-share mid/small caps carry deeper systematic
drawdowns (2015/2018 crashes) and lower risk-adjusted returns; more names re-sample a *worse* beta.
Quality/diversification (which only cured concentration on HS300) make CSI500 worse still (-70%).
**Verdict: stay on HS300; CSI500 expansion is rejected.**

## A7 — rebalance cadence & combined universe: both confirm the deployed config

Two axes flagged as untested (`scripts/cadence_universe_study.py`, HS300 value+rev, 2015-2025):

**Cadence (top-10, deployed signal):** monthly is the peak at every tier. Quarterly cuts cost ~⅔
(43 vs 131 rebalances) but CAGR drops more -- the 1-month reversal goes stale -- so Calmar is lower
(0.21-0.23 vs monthly 0.26-0.32), even at ¥10k where cost matters most. Weekly churns (562
rebalances, ~4× cost) for no gain (Calmar 0.04-0.27). **Monthly is optimal; the a-priori expectation
that quarterly helps small accounts is refuted.**

**Combined HS300 ∪ CSI500 (1552 names, monthly, price limit ON + ST filtered):** dilutes -- combined
top-10 +2.1% / -60.2% / Calmar 0.03; top-30 +6.4% / -39.6% / 0.16; both far below HS300-alone
(0.32). The value screen reaches into riskier small-cap deep-value, deepening drawdown. **Do not
mix; HS300-alone is best** (consistent with A6).

## A8 — IF index-futures short hedge: REJECTED (the −33% is value-style, not market-beta)

The lever most likely to cut the −33% drawdown: overlay a short CSI 300 index futures (IF) position
on the long book to neutralize market beta. Built faithfully (`research/backtest/hedge.py`: integer
contracts, ¥300/pt, roll/basis carry swept 0-4%/yr; `scripts/index_hedge_study.py`). It does not work --
it makes the drawdown worse at every ratio:

| hedge ratio (cost 2%/yr) | CAGR  | maxDD  | Calmar | ann.vol |
|--------------------------|------:|-------:|-------:|--------:|
| 0.00 (unhedged)          | +10.5%| -33.1% | 0.32   | 20.4%   |
| 0.50                     | +7.9% | -41.6% | 0.19   | 17.5%   |
| 1.00 (full notional)     | +4.7% | -57.0% | 0.08   | 21.3%   |

**Diagnosis (the clean test):** the book's beta to HS300 is only **0.66** with **R² ≈ 0.49** -- it is
low-beta (banks/defensive) and only half its variance is market-driven. A *perfect* beta-hedge leaves
a residual whose maxDD is **−36.3%, worse than the unhedged −33.1%** -- the market beta *cushions* the
worst drawdown, it does not cause it.

The two drawdown episodes make the mechanism precise (an earlier, looser account was incorrect;
corrected in adversarial review):
  - The **unhedged −33% maxDD is the 2015 crash** (2015-07→08), where HS300 *fell* −22.5% alongside the
    book (corr ~0.73). They crash together, so a short hedge would have helped *this* window.
  - The hedge nonetheless worsens the *overall* maxDD because shorting the index opens a new, deeper
    drawdown in the **2020-2021** stretch, where the index was ~flat/up (+1.6%) while value lagged -- a
    relative/style underperformance the short turns into an absolute loss.
  - Decisive floor: even a costless **look-ahead oracle** that shorts only during the exact 2015 crash
    cannot push maxDD below **~−31.3%** -- removing 2015 just promotes the 2020-2021 (index-flat)
    drawdown to be the new maximum. No index short can touch a drawdown that occurs with the index flat.

An index short reduces *volatility* (minimized near beta-neutral, 20.4%→16.5%) but cannot cut the
*maximum drawdown*, which is an **intrinsic value-style tail**, not a beta event. (Aside: 1 IF ≈ ¥1.4M
notional, so a clean hedge requires ≥~¥5M, which small accounts cannot reach.) Adversarial check: no realizable
IF/IC/IH short -- any ratio, time-varying-beta, or regime/vol-conditional -- cuts the −33% without
look-ahead; every tradeable ratio worsens it monotonically (h=0.5 −41.6%, h=1 −57.0%).

To hedge a value-style drawdown one must hedge the style (long cheap / short expensive = long-short),
not the market -- a structurally different strategy, and A-share securities lending is costly,
scarce, and constrained, so it is not a costless option.

## A9 — per-name stop-loss / take-profit: REJECTED

The deployed book has no price stops; tested as an ablation (full daily-loop, no look-ahead, vs the
no-stop baseline Calmar ~0.32 / maxDD −33%):

| overlay        | best setting | ΔCalmar | maxDD effect |
|----------------|-------------|--------:|--------------|
| stop-loss      | −10/−15/−20/−25% | −0.085 to −0.106 (all negative) | none cut it; −15%/−20% **deepen** it (−6pp, −4pp) |
| take-profit    | +15/+20/+30/+50% | +0.04 fragile | non-monotone, sign-flips across sub-periods |

**Stops harm the strategy** -- they dump the oversold names a value/reversal book is designed to retain (same
class as A1 portfolio-timing). Take-profit's small bump is single-event (2015-08) regime luck that flips
sign out-of-sample. **Keep the book as-is: no price stops, no take-profit.** (The catastrophe exits that
do belong -- delisting force-liquidation, ST filter -- are already in the engine.)

## CSI500-native factors (follow-up to A6): a real signal, but unharvestable long-only

Re-tested CSI500 with small-cap-native factors (not the HS300-tuned value). A genuine small-cap
predictor cluster exists at the IC level -- **low turnover-volatility (turnstd20: mean IC +0.084,
t=6.15, hit 74%, positive in every 2y sub-period)**, plus low-turnover / illiquidity / low-vol. But it
is a **left-tail short signal** (it flags names to avoid): long-only it still backtests to Calmar ~0.20
/ −44% with 0.75 correlation to the HS300 book (no diversification). It is only harvestable in a
long-short book -- the same securities-lending-gated constraint as A8. **Standalone long-only CSI500 stays rejected under
any factor sophistication.**

## Intraday IF (separate line): no edge, data-gated -- see [intraday.md](intraday.md)

A first intraday IF-futures probe found no validated edge (the two positives were a look-ahead artifact
and an overfit spike), and the free Sina minute history (~1yr) is too shallow for any verdict. Parked
pending a multi-year data-accumulation effort. It is a *separate* strategy domain, not a drawdown lever.

### Where this leaves the drawdown

**Every lever has now been tested -- selection (A3/A4/quality), weighting (A2), factor breadth (B),
market timing (A1), universe (A6/A7), cadence (A7), index hedge (A8), price stops/take-profit (A9),
CSI500-native factors, and a first intraday line -- and they converge on one answer: the deployed HS300
value + light 1-month-reversal, monthly, top-10 (Calmar ~0.32, maxDD −33%) is the best long-only
A-share strategy reachable, and there is no new tradeable long-only edge to add.** The −33% is the
intrinsic, value-style drawdown of harvesting the value premium long-only; no selection / weighting /
universe / cadence / stop / index hedge removes it. The only structurally different option is a true
long-short book (hedge the style / harvest the small-cap short signal), gated by costly, scarce,
retail-infeasible A-share securities lending. The deployable frontier for this account is now operational -- run the
paper record forward to make the 0.32 Calmar credible out-of-sample -- and capacity-aware, not another
drawdown lever.
