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

mean IC **+0.024** (t +1.47) — **BELOW low_vol alone (+0.067)**. Stacking the weak
factors (momentum/reversal) into the model diluted the signal rather than improving it.

## PIT backtest, top-10 equal weight (CAGR / max drawdown)

| strategy    | 100k          | 1,000,000     |
|-------------|--------------:|--------------:|
| ep (1/PE)   | +9.2% / -33%  | +9.5% / -34%  |
| ML-5factor  | +3.1% / -47%  | +5.7% / -60%  |
| low_vol     | -0.3% / -47%  | -0.2% / -47%  |
| momentum    | -7.0% / -77%  | -8.0% / -88%  |

## Lessons (the point of building this rigorously)

1. **The single value factor (earnings yield) is the strongest AND lowest-drawdown — it
   beats the LightGBM combiner.** Complexity != edge.
2. **High IC != high top-decile return.** low_vol had the best IC but a flat backtest; its
   edge is risk-adjusted/full-distribution, not concentrated in the top names. ep had a
   lower IC but the best CAGR — value's payoff concentrates in the cheapest names.
3. Naive ML stacking diluted a strong factor with noisy ones. A value-tilted or
   IC-weighted combination, or dropping the weak factors, is the obvious next iteration.
4. Even the best (ep, -33% maxDD) is **not deployable as-is** — drawdown control (vol
   targeting, sector caps, a market filter) is the next real problem, not more factors.

## Caveats

Equal-weight top-10 is crude; 涨跌停 no-fill is not modeled in this engine (cross-checked
via RQAlpha); costs ~ default 万2.5/万5; a single market regime/period; turnover is not
penalized. Treat magnitudes as indicative, the RANKING (value > ML-stack > low-vol >
momentum) as the robust takeaway.
