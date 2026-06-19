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

The filter HURT both axes: it gutted the return (9.2% -> 0.7%) and even worsened max
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

### The survivorship leak this caught (the real lesson)

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
−33% is systematic. Cutting the drawdown itself still needs the *universe* dimension
(CSI 500/1000), sector-neutralization, or the deferred size factor -- all pending new data.
