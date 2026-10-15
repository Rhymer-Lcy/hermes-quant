# Engine cross-validation

> **Scope (corrected 2026-10-15).** This page records one historical cross-check, run on
> 2026-06-19, of naive 20-day momentum. It is **not** a validation of the currently
> paper-traded strategy (value with a light 1-month-reversal tilt, `DEPLOYED` in
> `src/hermes/live/strategy.py`), which has never been run through RQAlpha. The hand-rolled
> engine has changed since 2026-06-19 and this comparison has not been rerun. The results below
> are preserved as recorded; the claims drawn from them are qualified under "Findings" and
> "Limitations".

The hand-rolled `momentum_portfolio_backtest` is a transparent teaching/feasibility
tool with documented simplifications (forward-adjusted levels used for lot-sizing;
price-limit no-fill not modeled). To test whether its survivorship-free result was an artifact
of those simplifications, the same strategy -- monthly equal-weight top-10 HS300 by 20-day
return, point-in-time membership -- was run on **RQAlpha**, an independent mature A-share engine
using RiceQuant's own price data and native dividends/splits, price-limit no-fill,
T+1, stamp tax, and 100-share lots.

Membership is held fixed (our BaoStock PIT snapshots feed both engines). The intended
differences were engine, price data and corporate-action handling; the two scripts also differ
in signal timing (see "Limitations"), so the comparison does not isolate the first three.

| capital   | hand-rolled PIT CAGR | RQAlpha PIT CAGR | absolute gap |
|----------:|---------------------:|-----------------:|-------------:|
| 100,000   | -7.0%                | -6.8%            | 0.2 pp       |
| 1,000,000 | -8.0%                | -5.2%            | 2.8 pp       |

Provenance: the CAGRs were recorded on 2026-06-19 (commit `cf5099b`) by copying each script's
printed summary into this page by hand. The 1,000,000 hand-rolled figure was first recorded as
-7.2% and was replaced by -8.0% in commit `156bbe4` (2026-06-20), whose message describes the
earlier value as stale; which run produced -7.2% is not recorded. RQAlpha's max drawdown and
Sharpe were printed at run time but not recorded, and neither engine has been rerun for this
page since.

Both engines give a negative CAGR at both capital tiers. Drawdowns were recorded for the
hand-rolled engine only (-77% at 100k, -88% at 1M; see [factor_research.md](factor_research.md)).

## Findings

1. **Agreement.** At 100k the two engines are 0.2 pp apart; at 1M they are 2.8 pp apart, with
   RQAlpha less negative. This is consistent with the hand-rolled result not being grossly
   wrong in sign or magnitude for this strategy and window. It does not establish trade-level
   fidelity: no fills were compared.
2. **Robustness.** The negative sign held across two engines and two price vendors, in one
   strategy, one window (2015-01-01 to 2025-12-31) and one run per engine.
3. **Bug 2** (forward-adjusted price levels feeding lot-sizing). RQAlpha sizes on raw prices
   with dividends and reaches the same sign. The comparison does not isolate the lot-sizing
   effect from the other differences, and the 1M gap is unattributed, so this check alone does
   not establish that the effect is second-order.
4. **Survivorship.** The ~27-30pp CAGR gap vs the survivorship-biased version (biased
   +20.4%/+21.5% vs PIT -7.0%/-8.0% at 100k/1M) is a hand-rolled result only; RQAlpha was run on
   the PIT universe alone.

## Limitations

- **Same-bar signal and execution in the RQAlpha script.** `scripts/rqalpha_crosscheck.py`
  registers `scheduler.run_monthly(rebalance, tradingday=1)` with no `time_rule`. At the pinned
  RQAlpha commit (`745d81c`), a daily backtest fires such a function on the bar event
  (`next_bar_` in `rqalpha/mod/rqalpha_mod_sys_scheduler/scheduler.py`); `history_bars` called
  there returns that day's bar (docstring in `rqalpha/apis/api_base.py`); and daily orders match
  at the current bar (`matching_type` default in
  `rqalpha/mod/rqalpha_mod_sys_simulation/__init__.py`). The script therefore ranks names on the
  close it then trades at. The hand-rolled engine ranks on the month-end close and trades at the
  next trading day's close. Both trade on the first trading day of the month; the ranking dates
  differ by one trading day. The impact on either tier has not been measured.
- **Residual gap not attributed.** Candidate causes, none quantified: RQAlpha's default
  commission and stamp-tax settings differ from the hand-rolled cost model; RQAlpha skips trades
  on price-limit days, which the hand-rolled engine does not (RQAlpha logs these at run time, and
  that log was not retained); the two use different price vendors; and the signal-timing
  difference above.
- **Summary statistics only.** Neither engine's trades were exported or reconciled, no
  numerical acceptance tolerance was set in advance, and the cross-check is not part of CI.

## Reproduce

```
# hand-rolled (with the hermes env active):
python scripts/survivorship_study.py
# RQAlpha (with the separate rqalpha env active; needs `rqalpha download-bundle` once):
python scripts/rqalpha_crosscheck.py
```

The `rqalpha` env is separate (RQAlpha's dependency set, kept out of `hermes`); it is
pinned in `requirements-rqalpha.lock.txt`. RQAlpha's free bundle has no PIT
`index_components` (that needs paid rqdatac), so the cross-check feeds our BaoStock
membership and converts codes (`sh.600000` -> `600000.XSHG`).
