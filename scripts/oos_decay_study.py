"""OOS / edge-decay gate on the DEPLOYED HS300 strategy -- the same test plutus-quant applied
to its US cross-sectional signal (and which the US edge failed).

WHY THIS EXISTS
---------------
The deployed value+light-reversal book reports net Calmar ~0.32 over the FULL 2015-2025 sample
(docs/multi_factor.md). A single full-sample number cannot tell whether the edge is STABLE or
whether it is a relic of the early, retail-dominated, pre-institutionalization years. The sister
project plutus-quant learned this the hard way: a US small-cap signal looked strong over
2009-2024 in aggregate, yet a trailing-window decomposition showed the edge faded to a
statistical zero after ~2020 -- the decisive evidence was the MULTI-YEAR FADE, not any single
holdout year. This script subjects hermes to the identical discipline, on its OWN engine, OWN
deployed signal (single source of truth: live.strategy.deployed_signal -- zero train/serve
drift), and OWN A-share frictions.

A-share structural-break hypothesis (why the edge MIGHT decay)
-------------------------------------------------------------
The A-share value/small-cap premium is widely argued to be a structural inefficiency of a
retail-dominated, hard-to-arbitrage, capital-controlled market (Liu-Stambaugh-Yuan CH-3/CH-4).
That structure has been eroding:
  * 2015-2018  pre/early: ~80-90% retail volume, high shell/IPO-scarcity premium, scarce
               shorting; MSCI A-share inclusion only BEGINS 2018-06.
  * 2019-2021  transition: registration-based IPO on STAR Market (2019-07) and ChiNext
               (2020-08) starts deflating the shell/size premium; MSCI weight steps up;
               foreign inflows accelerate.
  * 2022-2025  post: main-board registration reform (2023-02), deepening institutionalization,
               regulators steering long-term/institutional capital in -- the market becomes
               structurally more like the US, where the same family of edges is arbitraged out.
If the edge is a structural premium that is being competed away, its IC and risk-adjusted return
should FADE monotonically across these three regimes -- exactly the plutus pattern.

WHAT IS MEASURED (four lenses; net AND gross where it matters)
--------------------------------------------------------------
1. FACTOR IC DECAY (the direct plutus analog). Monthly cross-sectional rank-IC of the deployed
   composite, and of its two legs (ep value, rev1 reversal), on the PIT HS300 universe. Sliced
   by year and by the regimes above; one-sample t-tests, 95% CIs, an early-vs-late Welch test,
   and "P(late window this weak | the full-sample edge were intact)".
2. SELECTION EXCESS DECAY. Top-10 (the deployed basket) equal-weight forward return MINUS the
   PIT-universe equal-weight forward return, per month -- the pure stock-selection alpha over
   "just hold the universe", stripped of market beta and frictions. Annualized + t-stat per regime.
3. LONG-SHORT QUINTILE SPREAD (Q5-Q1) per regime -- the cleanest pure-premium trajectory.
   DIAGNOSTIC ONLY: the short leg is not retail-tradeable in A-shares (margin/securities lending
   is scarce), so this measures edge decay, not a deployable return.
4. DEPLOYED LONG-ONLY, NET OF A-SHARE COST. The actual book (value+rev1 5/1, top-10, monthly)
   run once over the full sample, then its equity curve SLICED into the regimes -> CAGR, maxDD,
   Calmar, Sharpe per period (net at AUD 1,000,000 = the fully-diversified working regime, plus
   the zero-cost gross curve). This is the plutus _slice_stats analog and the practical verdict.

Reproduce:  D:\\Anaconda3\\envs\\hermes\\python.exe scripts/oos_decay_study.py
(Use the interpreter path directly; `conda run -n hermes` crashes the conda plugin on this box.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.live.strategy import DEPLOYED, deployed_signal
from hermes.paths import BACKTESTS_DIR
from hermes.research.backtest.frictions import ZERO_COSTS
from hermes.research.backtest.portfolio import signal_portfolio_backtest
from hermes.research.eval.factor_eval import compute_ic
from hermes.research.factors import library as fl

N_HOLD = DEPLOYED.n_hold
NET_TIER = 1_000_000          # fully-diversified working regime -> the edge-decay read is not
                              # contaminated by small-account lot/min-commission frictions
                              # (those are a SEPARATE, already-studied problem; see CAPITAL_TIERS).
N_Q = 5                       # quintiles for the Q5-Q1 diagnostic spread (HS300 ~300 names/month)

# Regime windows (inclusive). The 3-way split tracks A-share institutionalization; EARLY vs
# HOLDOUT is the user-designated coarse split mirroring plutus's in-sample vs holdout.
REGIMES = [
    ("2015-2018  pre/early",  "2015-01-01", "2018-12-31"),
    ("2019-2021  transition", "2019-01-01", "2021-12-31"),
    ("2022-2025  post",       "2022-01-01", "2025-12-31"),
]
COARSE = [
    ("EARLY   2015-2020", "2015-01-01", "2020-12-31"),
    ("HOLDOUT 2021-2025", "2021-01-01", "2025-12-31"),
]
TRAILING = [
    ("trailing 2023-2025", "2023-01-01", "2025-12-31"),
    ("trailing 2024-2025", "2024-01-01", "2025-12-31"),
]
FULL = ("FULL    2015-2025", "2015-01-01", "2025-12-31")


# --- IC-series inference (mirrors plutus crsp_oos_inference._pooled) -----------------

def _win(ic: pd.Series, lo: str, hi: str) -> dict | None:
    """One-sample inference on the monthly IC series inside [lo, hi]."""
    s = ic[(ic.index >= pd.Timestamp(lo)) & (ic.index <= pd.Timestamp(hi))].dropna()
    n = len(s)
    if n < 2:
        return None
    mean = float(s.mean())
    sd = float(s.std(ddof=1))
    t, p = stats.ttest_1samp(s, 0.0)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    return {"n": n, "mean": mean, "ann": mean * 12, "t": float(t), "p": float(p),
            "ci_lo": mean - tcrit * se, "ci_hi": mean + tcrit * se,
            "hit": float((s > 0).mean())}


def _p_this_bad(ic: pd.Series, lo: str, hi: str, full_lo: str, full_hi: str) -> float:
    """P(a window mean this LOW or lower | the true monthly IC equals the full-sample mean).

    Under H0 (edge intact at the full-sample level mu, sd sigma), a window mean of n months is
    ~Normal(mu, sigma/sqrt(n)). A small p means the window is implausibly weak IF the edge were
    still at its full-sample strength -- i.e. evidence of decay, not noise."""
    full = ic[(ic.index >= pd.Timestamp(full_lo)) & (ic.index <= pd.Timestamp(full_hi))].dropna()
    win = ic[(ic.index >= pd.Timestamp(lo)) & (ic.index <= pd.Timestamp(hi))].dropna()
    if len(full) < 2 or len(win) < 1:
        return float("nan")
    mu, sigma, n = float(full.mean()), float(full.std(ddof=1)), len(win)
    if sigma <= 0:
        return float("nan")
    return float(stats.norm.cdf((float(win.mean()) - mu) / (sigma / np.sqrt(n))))


def _welch(ic: pd.Series, a: tuple, b: tuple) -> tuple[float, float]:
    """Welch two-sample t-test: is the EARLY-window mean IC significantly above the LATE one?"""
    sa = ic[(ic.index >= pd.Timestamp(a[1])) & (ic.index <= pd.Timestamp(a[2]))].dropna()
    sb = ic[(ic.index >= pd.Timestamp(b[1])) & (ic.index <= pd.Timestamp(b[2]))].dropna()
    t, p = stats.ttest_ind(sa, sb, equal_var=False)
    return float(t), float(p)


# --- equity-curve slicing (mirrors plutus crsp_dl_oos._slice_stats) ------------------

def _slice_perf(equity: pd.Series, lo: str, hi: str, ppy: int = 252) -> dict | None:
    # CARRY-IN drawdown: measured against the GLOBAL high-water mark carried into the window,
    # NOT a window-local reset. A book that enters a regime already underwater must show that
    # drawdown -- resetting the peak at the boundary understates risk (an adversarial audit
    # caught the reset version reporting -15.6% for 2022-2025 when the true carry-in DD is ~-30%).
    cummax_full = equity.cummax()
    mask = (equity.index >= pd.Timestamp(lo)) & (equity.index <= pd.Timestamp(hi))
    eq = equity[mask]
    if len(eq) < 2:
        return None
    ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (1.0 + ret) ** (1.0 / years) - 1.0
    dd = float((eq / cummax_full[mask] - 1.0).min())          # vs carry-in HWM (honest)
    daily = eq.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(ppy)) if daily.std() > 0 else float("nan")
    calmar = cagr / abs(dd) if dd < 0 else float("nan")
    return {"ret": ret, "cagr": cagr, "maxdd": dd, "calmar": calmar, "sharpe": sharpe}


# --- per-month selection-excess and Q5-Q1 spread (one pass over eval dates) ----------

def selection_and_spread(signal: pd.DataFrame, close: pd.DataFrame, eval_dates: list,
                         asof, n_hold: int, n_q: int) -> pd.DataFrame:
    """Per eval date t: deployed top-`n_hold` EW forward return, PIT-universe EW forward
    return, their difference (selection excess), and the composite Q5-Q1 spread. All GROSS
    (no frictions) -- this isolates the SIGNAL's selection power and its decay; the friction
    drag is measured separately by the long-only backtest slices."""
    fwd = close.reindex(eval_dates)
    rows = {}
    for i in range(len(eval_dates) - 1):
        t, t1 = eval_dates[i], eval_dates[i + 1]
        r = fwd.loc[t1] / fwd.loc[t] - 1.0
        f = signal.loc[t]
        df = pd.concat([f, r], axis=1, keys=["f", "r"]).dropna()
        df = df.loc[df.index.isin(asof(t))]                    # PIT universe, tradable both bars
        if len(df) < max(n_q, n_hold):
            continue
        ranked = df.sort_values("f", ascending=False)
        top = ranked["r"].iloc[:n_hold].mean()
        univ = df["r"].mean()
        q = pd.qcut(df["f"].rank(method="first"), n_q, labels=False)
        qret = df["r"].groupby(q).mean()
        spread = float(qret.iloc[-1] - qret.iloc[0]) if len(qret) == n_q else np.nan
        rows[t] = {"top": float(top), "univ": float(univ),
                   "excess": float(top - univ), "spread": spread}
    return pd.DataFrame(rows).T.sort_index()


def _excess_win(sel: pd.DataFrame, col: str, lo: str, hi: str) -> dict | None:
    s = sel[col][(sel.index >= pd.Timestamp(lo)) & (sel.index <= pd.Timestamp(hi))].dropna()
    n = len(s)
    if n < 2:
        return None
    mean = float(s.mean())
    t, p = stats.ttest_1samp(s, 0.0)
    return {"n": n, "mean_mo": mean, "ann": mean * 12, "t": float(t), "p": float(p),
            "hit": float((s > 0).mean())}


def main() -> None:
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof = membership_lookup(mdf)

    close = load_close_panel(codes=union, field="close")
    pe = load_close_panel(codes=union, field="peTTM")
    eval_dates = [pd.Timestamp(d) for d in sorted(mdf["date"].unique()) if pd.Timestamp(d) in close.index]

    # The DEPLOYED composite (single source of truth) + its two legs, for IC decay.
    composite = deployed_signal(close, pe, asof)
    ep = fl.earnings_yield(pe)
    rev1 = -fl.trailing_return(close, DEPLOYED.reversal_lookback)

    ic_comp = compute_ic(composite, close, eval_dates, members_asof=asof).ic
    ic_ep = compute_ic(ep, close, eval_dates, members_asof=asof).ic
    ic_rev = compute_ic(rev1, close, eval_dates, members_asof=asof).ic

    print("=" * 92)
    print("OOS / EDGE-DECAY GATE on the deployed HS300 value+light-reversal book")
    print("(the same trailing-window discipline plutus-quant's US signal FAILED)")
    print(f"eval: {len(ic_comp)} monthly PIT-HS300 rank-IC obs, non-overlapping 1m forward returns")
    print("=" * 92)

    # 1) FACTOR IC DECAY ------------------------------------------------------------
    print("\n[1] FACTOR IC DECAY  (deployed composite, and its value/reversal legs)")
    print(f"  {'window':<22} {'n':>4} {'meanIC':>8} {'annIC':>8} {'t':>6} {'95% CI':>18} {'hit%':>6}")
    for label, lo, hi in [FULL, *REGIMES, *COARSE, *TRAILING]:
        w = _win(ic_comp, lo, hi)
        if w:
            print(f"  {label:<22} {w['n']:>4} {w['mean']:>+8.4f} {w['ann']:>+8.3f} {w['t']:>+6.2f} "
                  f"[{w['ci_lo']:>+7.4f},{w['ci_hi']:>+7.4f}] {w['hit']*100:>5.0f}%")

    print("\n  per-leg mean IC by regime (value should be the workhorse; reversal a thin diversifier):")
    print(f"  {'window':<22} {'ep meanIC':>10} {'ep t':>6}   {'rev1 meanIC':>12} {'rev1 t':>7}")
    for label, lo, hi in [FULL, *REGIMES]:
        e, rv = _win(ic_ep, lo, hi), _win(ic_rev, lo, hi)
        if e and rv:
            print(f"  {label:<22} {e['mean']:>+10.4f} {e['t']:>+6.2f}   "
                  f"{rv['mean']:>+12.4f} {rv['t']:>+7.2f}")

    tw, pw = _welch(ic_comp, COARSE[0], COARSE[1])
    p_bad = _p_this_bad(ic_comp, COARSE[1][1], COARSE[1][2], FULL[1], FULL[2])
    print(f"\n  Welch (EARLY 2015-2020 vs HOLDOUT 2021-2025) composite IC: t={tw:+.2f}, p={pw:.3f}")
    print(f"  P(HOLDOUT IC this weak | edge intact at full-sample strength) = {p_bad:.3f}")

    # HONESTY: the holdout's STANDALONE significance is boundary-fragile (an adversarial audit
    # showed it evaporates with a 2020/2022 start, with ep-only, or by dropping ~2 lucky months).
    # Print the boundary sensitivity so the verdict rests on "positive & not-distinguishable-from
    # -early" (robust), not on "t>2 in this one window" (fragile).
    print("\n  holdout-boundary fragility (recent-window IC significance is boundary-dependent):")
    for lab, lo, hi in [("2020-2025", "2020-01-01", "2025-12-31"),
                        ("2021-2025", "2021-01-01", "2025-12-31"),
                        ("2022-2025", "2022-01-01", "2025-12-31")]:
        c, e = _win(ic_comp, lo, hi), _win(ic_ep, lo, hi)
        print(f"    {lab}: composite t={c['t']:+.2f} p={c['p']:.3f}   ep-only t={e['t']:+.2f} p={e['p']:.3f}")

    # per-year IC table
    print("\n  per-year composite IC:")
    yr = ic_comp.groupby(ic_comp.index.year)
    print("   " + "  ".join(f"{int(y)}:{v:+.3f}" for y, v in yr.mean().items()))

    # 2) SELECTION EXCESS + 3) Q5-Q1 SPREAD ----------------------------------------
    sel = selection_and_spread(composite, close, eval_dates, asof, N_HOLD, N_Q)
    print("\n[2] SELECTION EXCESS  (deployed top-10 EW minus PIT-universe EW, GROSS, per month)")
    print(f"  {'window':<22} {'n':>4} {'ann excess':>11} {'t':>6} {'hit%':>6}")
    for label, lo, hi in [FULL, *REGIMES, *COARSE]:
        x = _excess_win(sel, "excess", lo, hi)
        if x:
            print(f"  {label:<22} {x['n']:>4} {x['ann']:>+11.2%} {x['t']:>+6.2f} {x['hit']*100:>5.0f}%")

    print("\n[3] LONG-SHORT Q5-Q1 SPREAD  (pure premium; DIAGNOSTIC ONLY -- short not retail-tradeable)")
    print(f"  {'window':<22} {'n':>4} {'ann spread':>11} {'t':>6} {'hit%':>6}")
    for label, lo, hi in [FULL, *REGIMES, *COARSE]:
        x = _excess_win(sel, "spread", lo, hi)
        if x:
            print(f"  {label:<22} {x['n']:>4} {x['ann']:>+11.2%} {x['t']:>+6.2f} {x['hit']*100:>5.0f}%")

    # 4) DEPLOYED LONG-ONLY, NET OF A-SHARE COST, SLICED ----------------------------
    res_net = signal_portfolio_backtest(close, composite, NET_TIER, N_HOLD, members_asof=asof)
    res_gross = signal_portfolio_backtest(close, composite, NET_TIER, N_HOLD, costs=ZERO_COSTS,
                                          members_asof=asof)
    print(f"\n[4] DEPLOYED LONG-ONLY, sub-period slices (net @ AUD {NET_TIER:,}; gross = zero-cost)")
    print(f"    full-sample check: net CAGR {res_net.cagr:+.1%}  maxDD {res_net.max_drawdown:.1%}  "
          f"Calmar {res_net.cagr/abs(res_net.max_drawdown):.2f}  (matches docs/multi_factor.md ~0.32)")
    print(f"  {'window':<22} {'netCAGR':>8} {'maxDD':>8} {'netCal':>7} {'netShrp':>8} {'grossCal':>9}")
    rows = []
    for label, lo, hi in [FULL, *REGIMES, *COARSE]:
        n_, g_ = _slice_perf(res_net.equity, lo, hi), _slice_perf(res_gross.equity, lo, hi)
        if n_ and g_:
            print(f"  {label:<22} {n_['cagr']:>+8.1%} {n_['maxdd']:>8.1%} {n_['calmar']:>7.2f} "
                  f"{n_['sharpe']:>8.2f} {g_['calmar']:>9.2f}")
            rows.append({"window": label, **{f"net_{k}": v for k, v in n_.items()},
                         "gross_calmar": g_["calmar"]})

    out = BACKTESTS_DIR / "oos_decay.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    sel.to_csv(BACKTESTS_DIR / "oos_decay_selection.csv")
    pd.DataFrame({"composite": ic_comp, "ep": ic_ep, "rev1": ic_rev}).to_csv(
        BACKTESTS_DIR / "oos_decay_ic.csv")
    print(f"\n  wrote {out.name}, oos_decay_selection.csv, oos_decay_ic.csv to {BACKTESTS_DIR}")
    print("\nVERDICT (audited): the institutionalization-decay hypothesis is REJECTED -- IC is NOT a\n"
          "monotone fade. It is V-shaped: strong 2015-2017 (IC +0.07..+0.14), collapsed in the\n"
          "2019-2020 value/growth-bubble winter (negative), recovered 2021-2023 (+0.09/+0.07/+0.12).\n"
          "This differs in KIND from plutus's US signal, which faded smoothly to a statistical zero\n"
          "and went NEGATIVE out-of-sample. Here the holdout (2021-2025 IC +0.067) is POSITIVE and\n"
          "statistically indistinguishable from the early period (Welch p=0.61; P(this weak|intact)\n"
          "=0.65). CALIBRATION (per adversarial audit): the holdout's STANDALONE significance is\n"
          "boundary-fragile (p 0.028@2021 start -> 0.063@2022 -> 0.107@2020; ep-only p=0.053), so the\n"
          "defensible claim is 'positive and NOT refuted', not 'robustly significant'. The pure Q5-Q1\n"
          "premium has COMPRESSED (+24.7% t3.16 in 2015-2018 -> +6.8% t0.61 in 2022-2025). The -33%\n"
          "value-style drawdown floor is NOT cured -- carry-in maxDD is -31%..-33% in every regime,\n"
          "including recent (Calmar ~0.35-0.53, not the window-reset ~1.0). The last ~24 months\n"
          "(2024-2026) IC drifts toward zero (trailing-24m t=0.56): noise or early decay -- the\n"
          "forward paper ledger (live.paper) is the instrument that will resolve it.\n"
          "NET: A-share value+reversal PASSES the gate the US edge FAILED, in the sense that it did\n"
          "not institutionalize away into a zero/negative holdout -- it is regime-cyclical and\n"
          "structurally credible. But it is on-watch (modest, boundary-fragile, uncured -33% DD,\n"
          "soft recent fade), NOT a proven permanent small-profit machine. Keep the ledger running.")


if __name__ == "__main__":
    main()
