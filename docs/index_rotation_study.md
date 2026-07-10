# Index momentum rotation (dual momentum): REJECTED — fragile, and the defensive gate fails

The most retail-operable strategy format that exists — rotate monthly among 3 broad-index ETFs
(HS300 / CSI500 / ChiNext) with a treasury-index defensive leg, by trailing momentum — tested with
pre-registered parameters and no tuning pass. Reproduce: `python scripts/index_rotation_study.py`.

Pre-registered before looking: lookbacks 126d and 252-21d (both standard in the momentum
literature, both reported), top-1 selection, monthly signal at month-end close with next-day
execution, 15 bps per switch. Price indices (no free ETF bars), so equity legs understate total
return by the dividend yield while the treasury index accrues coupons — the bias favors the
defensive leg, making a *positive* equity-rotation read conservative and a negative one mild.

## Result (2015-2025, net of switch costs)

| book | CAGR | maxDD | Calmar | Sharpe |
|---|---:|---:|---:|---:|
| rotation, 6m momentum | +2.0% | -56.2% | 0.04 | 0.20 |
| rotation, 12-1 momentum | +6.6% | -46.5% | 0.14 | 0.40 |
| B&H HS300 | +2.5% | -46.7% | 0.05 | 0.23 |
| B&H CSI500 | +3.1% | -65.2% | 0.05 | 0.25 |
| B&H ChiNext | +7.3% | -69.7% | 0.11 | 0.39 |
| B&H treasury index | +4.0% | -1.0% | 3.94 | 5.83 |
| B&H equity equal-weight | +4.7% | -59.9% | 0.08 | 0.32 |

## Verdict: REJECTED, on three grounds

1. **Lookback fragility.** The two pre-registered variants disagree wildly (Calmar 0.04 vs 0.14).
   An edge that lives or dies on the lookback choice is a regime artifact, not a harvestable
   premium; picking 12-1 after seeing this table would be in-sample selection.
2. **The defensive gate does not defend.** Dual momentum's selling point is that the bond leg
   truncates crashes. Here the book still drew down -46.5% (12-1) and -56.2% (6m) — monthly
   momentum reacts too slowly to A-share crash dynamics (the same sharp, V-shaped moves that broke
   A1's 200-day-MA filter on the stock book). It held the treasury leg 39% of the time and still
   caught the crashes.
3. **Dominated by what the repo already runs.** The best rotation Calmar (0.14) is under half the
   deployed HS300 book's 0.32 with a DEEPER drawdown, and it is not a diversifier: its equity legs
   are the same beta the book already carries.

Consistent with A1 (docs/risk_control.md): timing A-share broad beta with slow-moving trend/momentum
signals fails in both the time-series (A1) and cross-sectional (here) form. No tuning pass follows —
per the pre-registration, the family is closed.
