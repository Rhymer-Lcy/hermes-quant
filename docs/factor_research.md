# Factor research: single-factor IC and a walk-forward ML combiner

All results are **survivorship-free** (point-in-time HS300 membership, incl. removed/
delisted names) and **no-look-ahead** (non-overlapping monthly IC; walk-forward LightGBM
trained only on the past). Equal-weight top-10, A-share frictions, 2015-2025 (ML OOS from
2017-07 after the training warm-up).

## Single-factor rank IC (monthly, PIT universe)

| factor                | mean IC | t-stat |
|-----------------------|--------:|-------:|
| low_vol 120d          | +0.067  | +2.92  |
| earnings yield (1/PE) | +0.054  | +2.67  |
| book yield (1/PB)     | +0.038  | +1.74  |
| reversal 1m           | +0.023  | +1.40  |
| momentum 6-1m         | +0.018  | +1.01  |

## Walk-forward LightGBM (5 factors) — out-of-sample

mean IC **+0.014** (t +0.81) — below low_vol alone (+0.067). Stacking the weak
factors (momentum/reversal) into the model diluted the signal rather than improving it.

> CORRECTED. This section originally reported IC +0.024 (t +1.47) and a stronger ML backtest
> (+3.1% / +5.7% CAGR). Those numbers predate the survivorship-leak fix recorded in
> [multi_factor.md](multi_factor.md) ("Fix shipped with this study"): `build_dataset` used to
> standardize over the survivorship-defined union rather than the then-current members. De-leaked,
> the ML combiner is weaker still — which strengthens, not weakens, this document's conclusion.

## PIT backtest, top-10 equal weight (CAGR / max drawdown)

| strategy    | 100k          | 1,000,000     |
|-------------|--------------:|--------------:|
| ep (1/PE)   | +9.2% / -33%  | +9.5% / -34%  |
| ML-5factor  | +1.4% / -49%  | +0.8% / -67%  |
| low_vol     | -0.3% / -47%  | -0.2% / -47%  |
| momentum    | -7.0% / -77%  | -8.0% / -88%  |

## Liquidity/turnover factors (the last untested family; `scripts/liquidity_factor_study.py`)

Battery from lake fields turn/amount, orientations fixed a-priori in the factor library:

| factor | mean IC | t-stat | hit% | top-10 long-only |
|---|---:|---:|---:|---|
| low_turnstd20 (stable turnover) | +0.078 | **+4.26** | 66% | +9.6% / -33.6% / Calmar 0.29 |
| low_turn20 (low turnover) | +0.073 | +3.68 | 64% | +8.1% / -35.9% / Calmar 0.23 |
| amihud20 (illiquidity) | -0.009 | -0.71 | 48% | (fails the IC gate) |

Turnover stability carries the strongest IC ever measured on this universe -- above low_vol
(+2.92) and value (+2.67) -- yet its top-10 basket still does not beat the deployed book
(Calmar 0.29 vs 0.32): the same shape as lesson 2 below (a full-distribution/defensive signal
whose strength is avoiding the churned left tail, not concentrating in a top basket). The
pre-registered blend sweep was then run and REJECTED it (see the turnover-stability section of
[multi_factor.md](multi_factor.md)): the best point's +0.008 net-Calmar edge sits below the
sweep's own noise floor, with no plateau, and the factor correlates 0.90 with low_vol -- the
redundancy cluster, not a diversifier. Amihud illiquidity is dead inside HS300 -- every member is
too liquid for the premium to differentiate.

## Lessons

1. **The single value factor (earnings yield) is the strongest and lowest-drawdown, and
   it beats the LightGBM combiner.** Added complexity did not improve the result.
2. High IC does not imply high top-decile return. low_vol had the best IC but a flat
   backtest; its edge is risk-adjusted/full-distribution, not concentrated in the top
   names. ep had a lower IC but the best CAGR — value's payoff concentrates in the
   cheapest names.
3. Naive ML stacking diluted a strong factor with noisy ones. A value-tilted or
   IC-weighted combination, or dropping the weak factors, is the next iteration to
   evaluate.
4. Even the best (ep, -33% maxDD) is not deployable as-is. Drawdown control (vol
   targeting, sector caps, a market filter) is the priority, not additional factors.

## Caveats

Equal-weight top-10 is crude; price-limit no-fill (no fill at the limit) is not modeled in
this engine (cross-checked via RQAlpha); costs ~ default 2.5 bps/5 bps; a single market
regime/period; turnover is not
penalized; and size (market cap) is not in this single-factor set (it was reconstructed
free from the lake and evaluated separately in [risk_control.md](risk_control.md) A4, and
rejected — a size tilt deepens the drawdown). Treat magnitudes as indicative and the
ranking (value > ML-stack >
low-vol > momentum) as the robust takeaway.
