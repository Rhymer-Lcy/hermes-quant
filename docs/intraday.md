# Intraday / futures research line (`hermes.intraday`)

A separate strategy domain from the monthly EOD equity book (different cadence, instruments,
microstructure). Originally proposed as a market hedge, but A8 disproved index hedging, so this
line stands on its own merits, not as a hedge. Mode: historical research and simulation only.

## First probe — intraday IF (CSI 300 stock index futures) signals: no edge (data too shallow to conclude)

Built 3 causal signals on IF minute bars (AKShare Sina, free): intraday time-series momentum,
opening-range breakout (ORB), overnight-gap fade. Backtested long/short with IF frictions
(commission ~0.115bps/side, slippage 1-2 ticks, ¥300/pt), full parameter sweep, net and gross.

**Verdict: no validated edge — do not trade any of them.** The only attractive results were artifacts of bias:
- Gap "continuation" (Sharpe 6.5-7.6) was a look-ahead artifact: a close-to-close P&L convention
  let the first-bar trade mechanically capture the very overnight gap it conditioned on. Pricing the
  entry from the achievable open collapses it to Sharpe ~0 (±1-2%). This is a latent first-bar pricing bug
  the production engine must avoid.
- ORB OR=2/60m (Sharpe 2.0) was a lone overfit spike: a +1-bar execution delay collapses it from
  160 trades/Sharpe 2.0 to 4 trades/Sharpe −0.9, with neighbors flat or negative. Momentum was net-negative
  almost everywhere.

The data is too shallow for any verdict: Sina caps at ~1023 bars → 60m gives only ~255 daily
observations over ~1 year (4 bars/session), 30m only ~128 days, over a single bull leg, on a
roll-spliced IF0 continuous contract. This is a clean pipeline/feasibility probe, not evidence.

### What this line actually needs first (data-depth gate)
No intraday verdict is possible until ≥2-3 years of minute bars accumulate. Sina only serves a recent
window, so the prerequisite is a daily incremental pull that accumulates IF minute bars forward
into a parquet lake (plus optional fixed-quarterly-contract stitching for finer 5m resolution with
documented roll gaps). Only after that: a look-ahead-free minute engine (`hermes.intraday.engine`)
whose first-of-day entries are priced from the open (never `close[t-1]`, the exact bug that produced
the spurious Sharpe ~7), with a built-in robustness battery (execution-delay test, split-half, open-vs-close pricing
check, deflated-Sharpe / reality-check across the whole sweep) that refuses a verdict below a
configured data-span minimum. Engine and data depth precede signals; the signals are not the deliverable.

Status: PARKED, with the data clock RUNNING. The daily accumulator (`scripts/accumulate_if_minute.py`,
task `hermes-if-accum`) has built the minute lake forward since 2026-06-23 for IF0, extended
2026-07-11 to IC0/IH0/IM0 (CSI 500 / SSE 50 / CSI 1000 main continuous) -- IC and IM are the
hedge legs any future mid/small-cap line would need, and IM's post-2022 history is the scarcest.
Research stays parked until the data-span gate above is met.
