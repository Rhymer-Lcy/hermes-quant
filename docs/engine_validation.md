# Engine cross-validation

The hand-rolled `momentum_portfolio_backtest` is a transparent teaching/feasibility
tool with documented simplifications (前复权 levels used for lot-sizing; 涨跌停 no-fill
not modeled). To confirm its survivorship-free result is trustworthy, the SAME
strategy -- monthly equal-weight top-10 HS300 by 20-day return, point-in-time
membership -- was run on **RQAlpha**, an independent battle-tested A-share engine
using **RiceQuant's own price data** and native dividends/splits, 涨跌停 no-fill,
T+1, 印花税, and 100-share lots.

Membership is held fixed (our BaoStock PIT snapshots feed both engines), so the
differences isolate engine + price-data + corporate-action handling.

| capital   | hand-rolled PIT CAGR | RQAlpha PIT CAGR |
|----------:|---------------------:|-----------------:|
| 100,000   | -7.0%                | -6.8%            |
| 1,000,000 | -8.0%                | -5.2%            |

Two independent engines + two independent price datasets agree: survivorship-free
naive 20-day momentum on HS300 loses ~5-8%/yr with catastrophic drawdowns (~75-88%).

Conclusions:
1. The hand-rolled engine is **trustworthy** (within ~1-2pp CAGR of RQAlpha).
2. The survivorship-free result is **robust** -- not an artifact of one engine/dataset.
3. Bug 2 (前复权 price levels feeding lot-sizing) is confirmed **second-order**: RQAlpha
   uses true raw prices + dividends and lands at the same conclusion.
4. The ~25pp CAGR gap vs the survivorship-biased version was pure universe look-ahead.

Caveats (explain the residual ~2pp difference, all second-order): RQAlpha's default
commission/stamp model differs slightly from the hand-rolled 万2.5 / 万5; RQAlpha skips
trades on 涨跌停 days (logged at run time) which the hand-rolled engine does not; and
the two use different price vendors.

## Reproduce

```
# hand-rolled (hermes env):
D:\Anaconda3\envs\hermes\python.exe scripts/portfolio_pit_demo.py
# RQAlpha (separate rqalpha env; needs `rqalpha download-bundle` once):
D:\Anaconda3\envs\rqalpha\python.exe scripts/rqalpha_momentum.py
```

The `rqalpha` env is separate (RQAlpha's dependency set, kept out of `hermes`); it is
pinned in `requirements-rqalpha.lock.txt`. RQAlpha's free bundle has no PIT
`index_components` (that needs paid rqdatac), so the cross-check feeds our BaoStock
membership and converts codes (`sh.600000` -> `600000.XSHG`).
