"""Does the calendar day of the monthly rebalance matter? Issue #21.

A7 settled the FREQUENCY (monthly beats weekly and quarterly at every tier). This asks the
orthogonal question: conditional on rebalancing monthly, does the within-month calendar day carry
a stable effect? All 31 days are enumerated, and the study is framed as a MULTIPLE-TESTING problem
-- enumerating 31 candidates over a finite sample guarantees a maximum, and the maximum is not
evidence. Design frozen in issue #21 before any sweep ran.

  mapping    target = min(D, days_in_month); execute on the first trading bar on or after the
             target WITHIN the month (else the month's last bar); signal on the preceding bar
  parity     D=1 must reproduce the canonical month-end engine bit-for-bit -- BLOCKING
  frozen     universe, signal, weights, n_hold, band, costs, stops: all unchanged; only D varies
  primary    CNY 1,000,000, 2015-01-01..2025-12-31
  stats      paired monthly net-return differences vs D=1, Newey-West HAC t, Holm across 30
             comparisons, gross agreement, plateau, regime stability, rank stability, year-block
             bootstrap
  verdict    thresholds A-H in the issue; ALL must hold, else REJECTED / NOT IDENTIFIED

Even a passing D does not modify DEPLOYED: the only permitted next step is a separate forward-test
registration. The D=1 paper ledger is never rewritten.

    python scripts/rebalance_timing_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps

from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.live.strategy import DEPLOYED, deployed_signal
from hermes.paths import BACKTESTS_DIR, FIGURES_DIR
from hermes.research.backtest.frictions import ZERO_COSTS
from hermes.research.backtest.portfolio import signal_portfolio_backtest
from hermes.research.backtest.schedule import (assert_no_lookahead, calendar_rebalance_schedule,
                                               duplicate_groups, month_end_schedule,
                                               nth_trading_day_schedule, schedule_hash,
                                               turnover_from_trades)

DAYS = list(range(1, 32))
BASELINE_D = 1
PRIMARY_TIER = 1_000_000
ROBUST_TIERS = [100_000, 5_000_000]
PRIMARY_WINDOW = ("2015-01-01", "2025-12-31")
SECONDARY_WINDOW = ("2016-01-01", "2025-12-31")
REGIMES = [("2015-01-01", "2018-12-31", "2015-2018"),
           ("2019-01-01", "2021-12-31", "2019-2021"),
           ("2022-01-01", "2025-12-31", "2022-2025")]
TRADING_DAYS_PER_YEAR = 243
ALPHA = 0.05
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260827          # fixed: Date.now()-free, and the run is reproducible


# --- metrics -------------------------------------------------------------------------

def sharpe(equity: pd.Series) -> float:
    r = equity.pct_change().dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR)) if sd > 0 else np.nan


def carry_in_drawdown(equity: pd.Series, lo: str, hi: str) -> float:
    """Sub-period drawdown against the GLOBAL running high-water mark.

    docs/oos_decay.md corrected exactly this: a window-local high-water mark flattered 2022-2025
    from -30.1% to -15.6%, because an account entering the window already down 24% gets its losses
    forgiven. The carried-in mark is the honest measure."""
    hwm = equity.cummax()
    seg, mark = equity.loc[lo:hi], hwm.loc[lo:hi]
    return float((seg / mark - 1.0).min()) if len(seg) else np.nan


def metrics(res, equity: pd.Series | None = None) -> dict:
    eq = res.equity if equity is None else equity
    dd = res.max_drawdown
    return {"total_return": res.total_return, "cagr": res.cagr, "max_drawdown": dd,
            "sharpe": sharpe(eq), "calmar": res.cagr / abs(dd) if dd else np.nan,
            "total_costs": res.total_costs, "turnover": turnover_from_trades(res.trades),
            "avg_names_held": res.avg_names_held, "n_rebalances": res.n_rebalances}


# --- sweep ---------------------------------------------------------------------------

def run_one(close, sig, asof, capital, sched, costs=None):
    return signal_portfolio_backtest(close, sig, capital, DEPLOYED.n_hold, costs=costs,
                                     members_asof=asof, rebalance_band=DEPLOYED.rebalance_band,
                                     collect_trades=True, schedule=sched)


def sweep(close, sig, asof, capital, label, gross=False):
    """One row per D. Returns (DataFrame, {D: equity series})."""
    rows, curves = [], {}
    scheds = {d: calendar_rebalance_schedule(close.index, d) for d in DAYS}
    groups = duplicate_groups(scheds)
    dup_of = {d: ds for ds in groups.values() for d in ds}
    for d in DAYS:
        assert_no_lookahead(scheds[d], close.index)          # invariant, every candidate
        res = run_one(close, sig, asof, capital, scheds[d], ZERO_COSTS if gross else None)
        curves[d] = res.equity
        rows.append({"D": d, "hash": schedule_hash(scheds[d]),
                     "dup_with": ",".join(str(x) for x in dup_of[d] if x != d) or "-",
                     **metrics(res)})
    df = pd.DataFrame(rows).set_index("D")
    base = df.loc[BASELINE_D]
    for c, nm in [("cagr", "dCAGR"), ("sharpe", "dSharpe"), ("calmar", "dCalmar"),
                  ("max_drawdown", "dMaxDD"), ("total_costs", "dCosts")]:
        df[nm] = df[c] - base[c]
    print(f"\n  [{label}] swept {len(df)} candidates, "
          f"{sum(1 for v in groups.values() if len(v) > 1)} duplicate group(s)")
    return df, curves


# --- statistics ----------------------------------------------------------------------

def hac_t(diff: pd.Series) -> tuple[float, float, int]:
    """Mean of a paired difference series with Newey-West HAC standard errors. Lag by the standard
    rule of thumb floor(4*(n/100)^(2/9)); monthly portfolio differences are mildly autocorrelated
    and an iid t would overstate significance."""
    y = diff.dropna().to_numpy(float)
    if len(y) < 8:
        return np.nan, np.nan, 0
    lag = int(np.floor(4 * (len(y) / 100) ** (2 / 9)))
    fit = sm.OLS(y, np.ones(len(y))).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    return float(fit.tvalues[0]), float(fit.pvalues[0]), lag


def holm(pvals: dict[int, float], alpha: float = ALPHA) -> pd.DataFrame:
    """Holm step-down over the 30 baseline comparisons. Returns the adjusted p and the reject flag
    in the order the procedure evaluates them."""
    items = sorted((p, d) for d, p in pvals.items() if np.isfinite(p))
    m, out, running = len(items), [], 0.0
    for i, (p, d) in enumerate(items):
        adj = min(1.0, max(running, p * (m - i)))     # enforce monotone step-down
        running = adj
        out.append({"D": d, "p_raw": p, "p_holm": adj, "reject": adj < alpha})
    return pd.DataFrame(out).set_index("D")


def monthly_returns(equity: pd.Series) -> pd.Series:
    m = equity.resample("ME").last().pct_change().dropna()
    m.index = m.index.to_period("M")
    return m


def plateau_runs(better: pd.Series) -> list[list[int]]:
    """Maximal runs of adjacent calendar days that all improve on the baseline. A real timing
    effect shows up in neighbours; a lone spike is read as mining noise."""
    runs, cur = [], []
    for d in DAYS:
        if bool(better.get(d, False)):
            cur.append(d)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def year_block_bootstrap(curves: dict[int, pd.Series], draws: int) -> pd.Series:
    """Resample whole CALENDAR YEARS with replacement and re-rank the candidates each draw.

    Whole-year blocks keep the within-year time structure (and the seasonal shape a calendar-day
    effect would live in) intact; an iid month bootstrap would destroy exactly the structure under
    test. Reports how often each D is the argmax -- a champion that wanders is winner instability."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    mr = {d: monthly_returns(c) for d, c in curves.items()}
    years = sorted({p.year for p in mr[BASELINE_D].index})
    by_year = {d: {y: s[[p.year == y for p in s.index]] for y in years} for d, s in mr.items()}
    wins = dict.fromkeys(DAYS, 0)
    for _ in range(draws):
        pick = rng.choice(years, size=len(years), replace=True)
        best_d, best_v = None, -np.inf
        for d in DAYS:
            v = float(np.sum([by_year[d][y].sum() for y in pick]))
            if v > best_v:
                best_d, best_v = d, v
        wins[best_d] += 1
    return pd.Series(wins).sort_values(ascending=False) / draws


# --- reporting -----------------------------------------------------------------------

def show(df: pd.DataFrame, cols: list[str], title: str, fmts: dict | None = None) -> None:
    d = df[cols].copy()
    for c, f in (fmts or {}).items():
        if c in d:
            d[c] = d[c].map(lambda v, f=f: f.format(v) if pd.notna(v) else "--")
    print(f"\n{title}")
    print(d.to_string())


PCT = "{:+.2%}"
NUM = "{:+.3f}"


def main() -> None:
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof = membership_lookup(mdf)
    lo, hi = PRIMARY_WINDOW
    close_all = load_close_panel(codes=union, end=hi)
    pe_all = load_close_panel(codes=union, field="peTTM", end=hi)
    close = close_all.loc[lo:hi]
    pe = pe_all.loc[lo:hi]
    sig = deployed_signal(close, pe, asof, DEPLOYED)

    # --- BLOCKING PARITY GATE ---------------------------------------------------
    print("=== PARITY GATE (blocking) ===")
    s1 = calendar_rebalance_schedule(close.index, BASELINE_D)
    assert s1 == month_end_schedule(close.index), "D=1 schedule != canonical month-end schedule"
    a = run_one(close, sig, asof, PRIMARY_TIER, None)
    b = run_one(close, sig, asof, PRIMARY_TIER, s1)
    assert a.equity.equals(b.equity), "D=1 equity differs from the canonical engine"
    assert len(a.trades) == len(b.trades) and a.n_rebalances == b.n_rebalances
    print(f"  bars {len(close.index)}, rebalances {a.n_rebalances}, trades {len(a.trades)}")
    print(f"  equity max abs diff {float((a.equity - b.equity).abs().max()):.3e}  -> PASS")
    print(f"  baseline: CAGR {a.cagr:+.2%}  maxDD {a.max_drawdown:.2%}  "
          f"Calmar {a.cagr / abs(a.max_drawdown):.3f}  costs {a.total_costs:,.0f}")

    # --- PRIMARY SWEEP ----------------------------------------------------------
    net, curves = sweep(close, sig, asof, PRIMARY_TIER, f"net {PRIMARY_TIER:,}")
    gross, curves_g = sweep(close, sig, asof, PRIMARY_TIER, f"gross {PRIMARY_TIER:,}", gross=True)

    show(net, ["hash", "dup_with", "n_rebalances", "total_return", "cagr", "max_drawdown",
               "sharpe", "calmar", "total_costs", "turnover", "avg_names_held"],
         "=== PRIMARY: all 31 candidates, natural order, NET (CNY 1,000,000, 2015-2025) ===",
         {"total_return": PCT, "cagr": PCT, "max_drawdown": PCT, "sharpe": NUM, "calmar": NUM,
          "total_costs": "{:,.0f}", "turnover": "{:,.0f}", "avg_names_held": "{:.2f}"})

    show(net, ["cagr", "dCAGR", "sharpe", "dSharpe", "calmar", "dCalmar", "max_drawdown",
               "dMaxDD", "dCosts"],
         "=== vs D=1 baseline (NET) ===",
         {"cagr": PCT, "dCAGR": PCT, "sharpe": NUM, "dSharpe": NUM, "calmar": NUM,
          "dCalmar": NUM, "max_drawdown": PCT, "dMaxDD": PCT, "dCosts": "{:+,.0f}"})

    show(net.sort_values("cagr", ascending=False), ["cagr", "dCAGR", "calmar", "sharpe",
                                                    "max_drawdown", "total_costs"],
         "=== ranked by CAGR (all 31, no truncation) ===",
         {"cagr": PCT, "dCAGR": PCT, "calmar": NUM, "sharpe": NUM, "max_drawdown": PCT,
          "total_costs": "{:,.0f}"})
    show(net.sort_values("calmar", ascending=False), ["calmar", "dCalmar", "cagr", "max_drawdown"],
         "=== ranked by Calmar (all 31) ===",
         {"calmar": NUM, "dCalmar": NUM, "cagr": PCT, "max_drawdown": PCT})
    show(gross, ["cagr", "calmar", "sharpe", "max_drawdown"],
         "=== GROSS (zero cost) -- is any net gain merely lower turnover? ===",
         {"cagr": PCT, "calmar": NUM, "sharpe": NUM, "max_drawdown": PCT})

    # --- PAIRED TESTS + HOLM ----------------------------------------------------
    base_m = monthly_returns(curves[BASELINE_D])
    base_mg = monthly_returns(curves_g[BASELINE_D])
    rows = []
    for d in DAYS:
        if d == BASELINE_D:
            continue
        diff = (monthly_returns(curves[d]) - base_m).dropna()
        dg = (monthly_returns(curves_g[d]) - base_mg).dropna()
        t, p, lag = hac_t(diff)
        tg, pg, _ = hac_t(dg)
        rows.append({"D": d, "n_months": len(diff), "mean_diff": diff.mean(), "hac_t": t,
                     "p_raw": p, "lag": lag, "gross_mean": dg.mean(), "gross_t": tg})
    paired = pd.DataFrame(rows).set_index("D")
    hol = holm(paired["p_raw"].to_dict())
    paired = paired.join(hol[["p_holm", "reject"]])
    show(paired, ["n_months", "mean_diff", "hac_t", "p_raw", "p_holm", "reject",
                  "gross_mean", "gross_t"],
         "=== paired monthly differences vs D=1, Newey-West HAC, Holm over 30 comparisons ===",
         {"mean_diff": "{:+.4%}", "hac_t": NUM, "p_raw": "{:.4f}", "p_holm": "{:.4f}",
          "gross_mean": "{:+.4%}", "gross_t": NUM})
    n_reject = int(paired["reject"].sum())
    print(f"\n  Holm rejections at alpha={ALPHA}: {n_reject} of {len(paired)}")

    # --- PLATEAU ----------------------------------------------------------------
    print("\n=== plateau check (contiguous runs of days beating D=1) ===")
    for tag, col in [("net CAGR", "dCAGR"), ("net Calmar", "dCalmar")]:
        runs = plateau_runs(net[col] > 0)
        longest = max((len(r) for r in runs), default=0)
        print(f"  {tag:<12}: {len(runs)} run(s), longest {longest} day(s)  "
              f"{[f'{r[0]}-{r[-1]}' for r in runs if len(r) >= 2][:6]}")

    # --- REGIMES (carry-in high-water mark) -------------------------------------
    print("\n=== regimes: CAGR and carry-in drawdown per candidate ===")
    reg_rows = []
    for d in DAYS:
        eq = curves[d]
        row = {"D": d}
        for r_lo, r_hi, name in REGIMES:
            seg = eq.loc[r_lo:r_hi]
            if len(seg) < 2:
                continue
            yrs = (seg.index[-1] - seg.index[0]).days / 365.25
            row[f"{name} CAGR"] = (seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1
            row[f"{name} DD"] = carry_in_drawdown(eq, r_lo, r_hi)
        reg_rows.append(row)
    reg = pd.DataFrame(reg_rows).set_index("D")
    for _lo, _hi, name in REGIMES:
        reg[f"{name} dCAGR"] = reg[f"{name} CAGR"] - reg.loc[BASELINE_D, f"{name} CAGR"]
    show(reg, [c for c in reg.columns if "CAGR" in c or "DD" in c],
         "  (dCAGR is vs D=1 within the same regime)",
         {c: PCT for c in reg.columns})

    # --- RANK STABILITY ---------------------------------------------------------
    # DISJOINT window pairs only. An earlier version also correlated NESTED pairs
    # (2015-2018 sits inside "first half"; 2022-2025 inside "second half"); a rho between a
    # window and a superset of itself is inflated by construction and carries no stability
    # information. The Spearman p is deliberately NOT reported either: spearmanr assumes
    # independent observations, but the 31 candidates are one strategy shifted a few days and
    # adjacent D share much of their execution calendar, so such a p is anti-conservative.
    print("\n=== rank stability: Spearman of the D-ranking, DISJOINT sub-periods only ===")
    halves = [(PRIMARY_WINDOW[0], "2020-06-30", "first half"),
              ("2020-07-01", PRIMARY_WINDOW[1], "second half")]
    rank_sets = {}
    for r_lo, r_hi, name in REGIMES + halves:
        vals = {}
        for d in DAYS:
            seg = curves[d].loc[r_lo:r_hi]
            vals[d] = seg.iloc[-1] / seg.iloc[0] - 1 if len(seg) > 1 else np.nan
        rank_sets[name] = pd.Series(vals).rank(ascending=False)
    disjoint = [("2015-2018", "2019-2021"), ("2015-2018", "2022-2025"),
                ("2019-2021", "2022-2025"), ("first half", "second half")]
    rhos = []
    for a_name, b_name in disjoint:
        rho = float(sps.spearmanr(rank_sets[a_name], rank_sets[b_name]).statistic)
        rhos.append(rho)
        print(f"  {a_name:<12} vs {b_name:<12}: rho {rho:+.3f}   (p omitted: candidates are not independent)")
    print(f"  mean rho over disjoint pairs: {np.mean(rhos):+.3f}  -> "
          f"{'stable' if np.mean(rhos) > 0.5 else 'RANKING UNSTABLE'}")

    # --- BOOTSTRAP --------------------------------------------------------------
    print(f"\n=== year-block bootstrap ({BOOTSTRAP_DRAWS} draws): how often is each D the winner? ===")
    wins = year_block_bootstrap(curves, BOOTSTRAP_DRAWS)
    print("  top 10 by win share:")
    for d, w in wins.head(10).items():
        print(f"    D={d:<3} {w:6.1%}")
    print(f"  distinct winners across draws: {(wins > 0).sum()} of 31; "
          f"max share {wins.max():.1%}; D=1 share {wins.get(BASELINE_D, 0):.1%}")

    # --- SECONDARY WINDOW + TIERS -----------------------------------------------
    s_lo, s_hi = SECONDARY_WINDOW
    # Signal computed on the FULL panel then sliced, not sliced then computed: the 20-day
    # reversal leg is otherwise NaN over the window's first 20 bars (429 signal cells),
    # discarding genuinely available prior data. Still strictly point-in-time -- the reversal
    # at the first bar reads only bars before it.
    c2 = close_all.loc[s_lo:s_hi]
    sig2 = deployed_signal(close_all, pe_all, asof, DEPLOYED).loc[s_lo:s_hi]
    sec, _ = sweep(c2, sig2, asof, PRIMARY_TIER, f"secondary {s_lo[:4]}-{s_hi[:4]}")
    show(sec, ["cagr", "dCAGR", "calmar", "dCalmar", "max_drawdown"],
         f"=== SECONDARY window {s_lo}..{s_hi} (complete decade) ===",
         {"cagr": PCT, "dCAGR": PCT, "calmar": NUM, "dCalmar": NUM, "max_drawdown": PCT})

    tiers = {}
    for cap in ROBUST_TIERS:
        t_df, _ = sweep(close, sig, asof, cap, f"robustness {cap:,}")
        tiers[cap] = t_df
        show(t_df, ["cagr", "dCAGR", "calmar", "dCalmar"],
             f"=== ROBUSTNESS tier CNY {cap:,} (may not drive the verdict) ===",
             {"cagr": PCT, "dCAGR": PCT, "calmar": NUM, "dCalmar": NUM})

    # --- SECONDARY DIAGNOSTIC ---------------------------------------------------
    print("\n=== SECONDARY DIAGNOSTIC (not part of the verdict): Nth trading day of the month ===")
    nth_rows = []
    for n_th in range(1, 24):
        sch = nth_trading_day_schedule(close.index, n_th)
        assert_no_lookahead(sch, close.index)
        res = run_one(close, sig, asof, PRIMARY_TIER, sch)
        nth_rows.append({"nth": n_th, **metrics(res)})
    nth = pd.DataFrame(nth_rows).set_index("nth")
    nth["dCAGR"] = nth["cagr"] - nth.loc[1, "cagr"]
    show(nth, ["cagr", "dCAGR", "calmar", "sharpe", "max_drawdown", "total_costs"],
         "  (vs the 1st trading day; separates a calendar-NUMBER effect from a POSITION effect)",
         {"cagr": PCT, "dCAGR": PCT, "calmar": NUM, "sharpe": NUM, "max_drawdown": PCT,
          "total_costs": "{:,.0f}"})

    # --- FIGURES ----------------------------------------------------------------
    _figures(net, gross, paired, reg, nth)

    # --- VERDICT ----------------------------------------------------------------
    _verdict(net, gross, paired, reg, sec, tiers, wins)

    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
    net.to_csv(BACKTESTS_DIR / "rebalance_timing_net.csv")
    gross.to_csv(BACKTESTS_DIR / "rebalance_timing_gross.csv")
    paired.to_csv(BACKTESTS_DIR / "rebalance_timing_paired.csv")
    reg.to_csv(BACKTESTS_DIR / "rebalance_timing_regimes.csv")
    sec.to_csv(BACKTESTS_DIR / "rebalance_timing_secondary.csv")
    nth.to_csv(BACKTESTS_DIR / "rebalance_timing_nth_trading_day.csv")
    wins.to_csv(BACKTESTS_DIR / "rebalance_timing_bootstrap.csv")
    print(f"\nsaved -> {BACKTESTS_DIR}")


def _figures(net, gross, paired, reg, nth) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(3, 2, figsize=(13, 11))
    b = net.loc[BASELINE_D]

    ax[0, 0].plot(net.index, net["cagr"] * 100, "o-", label="net")
    ax[0, 0].plot(gross.index, gross["cagr"] * 100, "s--", alpha=.6, label="gross")
    ax[0, 0].axhline(b["cagr"] * 100, color="r", ls=":", label="D=1 net")
    ax[0, 0].set_title("CAGR by calendar day")
    ax[0, 0].set_xlabel("D")
    ax[0, 0].legend()

    ax[0, 1].plot(net.index, net["sharpe"], "o-", label="Sharpe")
    ax[0, 1].plot(net.index, net["calmar"], "s-", label="Calmar")
    ax[0, 1].axhline(b["sharpe"], color="r", ls=":")
    ax[0, 1].axhline(b["calmar"], color="r", ls=":")
    ax[0, 1].set_title("Risk-adjusted by calendar day")
    ax[0, 1].set_xlabel("D")
    ax[0, 1].legend()

    ax[1, 0].plot(net.index, net["max_drawdown"] * 100, "o-")
    ax[1, 0].axhline(b["max_drawdown"] * 100, color="r", ls=":")
    ax[1, 0].set_title("max drawdown by calendar day")
    ax[1, 0].set_xlabel("D")

    ax2 = ax[1, 1].twinx()
    ax[1, 1].plot(net.index, net["turnover"] / 1e6, "o-", color="tab:blue")
    ax2.plot(net.index, net["total_costs"] / 1e3, "s--", color="tab:orange")
    ax[1, 1].set_ylabel("turnover, CNY m", color="tab:blue")
    ax2.set_ylabel("costs, CNY k", color="tab:orange")
    ax[1, 1].set_title("turnover and cost by calendar day")
    ax[1, 1].set_xlabel("D")

    for _lo, _hi, name in REGIMES:
        ax[2, 0].plot(reg.index, reg[f"{name} dCAGR"] * 100, "o-", label=name)
    ax[2, 0].axhline(0, color="k", lw=.8)
    ax[2, 0].set_title("dCAGR vs D=1, by regime")
    ax[2, 0].set_xlabel("D")
    ax[2, 0].legend()

    ax[2, 1].plot(nth.index, nth["cagr"] * 100, "o-")
    ax[2, 1].set_title("DIAGNOSTIC: CAGR by Nth trading day")
    ax[2, 1].set_xlabel("Nth trading day")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rebalance_timing.png", dpi=130)
    plt.close(fig)
    print(f"\nfigures -> {FIGURES_DIR / 'rebalance_timing.png'}")


def gate_table(net, gross, paired, reg, sec, tiers) -> pd.DataFrame:
    """The eight pre-registered gates, evaluated PER CANDIDATE DAY.

    Every gate must be a statement about the SAME day, or the conjunction certifies nothing: an
    earlier version tested A as "some day is significant" while C-H were read at the CAGR champion,
    which could print CONFIRMED for a champion whose own Holm p was 1.0. D=1 is the baseline and is
    not a candidate against itself.
    """
    runs = plateau_runs(net["dCAGR"] > 0)
    rows = []
    for d in DAYS:
        if d == BASELINE_D:
            continue
        rows.append({
            "D": d,
            # A: this day's own net improvement survives Holm over the 30 comparisons
            "A_holm": bool(paired.loc[d, "reject"]) and float(paired.loc[d, "mean_diff"]) > 0,
            # B: gross agrees -- read from the ZERO-COST sweep, so a net-only gain that is merely
            #    lower turnover cannot pass. The paired gross mean must agree too.
            "B_gross": float(gross.loc[d, "dCAGR"]) > 0 and float(paired.loc[d, "gross_mean"]) > 0,
            # C: this day sits inside a contiguous run of >= 3 days that all beat the baseline
            "C_plateau": any(len(r) >= 3 and d in r for r in runs),
            "D_regimes": all(float(reg.loc[d, f"{nm} dCAGR"]) > 0 for _l, _h, nm in REGIMES),
            "E_decade": float(sec.loc[d, "dCAGR"]) > 0,
            "F_riskadj": float(net.loc[d, "dSharpe"]) > 0 or float(net.loc[d, "dCalmar"]) > 0,
            "G_drawdown": float(net.loc[d, "dMaxDD"]) > -0.02,
            "H_tiers": float(net.loc[d, "dCAGR"]) > 0
                       and all(float(t.loc[d, "dCAGR"]) > 0 for t in tiers.values()),
        })
    tab = pd.DataFrame(rows).set_index("D")
    tab["n_pass"] = tab.sum(axis=1)
    tab["ALL_EIGHT"] = tab["n_pass"] == 8
    return tab


def _verdict(net, gross, paired, reg, sec, tiers, wins) -> None:
    print("\n" + "=" * 78)
    print("=== PRE-REGISTERED VERDICT (thresholds A-H frozen in issue #21) ===")

    champ = net["cagr"].idxmax()
    print(f"\n  historical CAGR maximum: D={champ}  "
          f"CAGR {net.loc[champ, 'cagr']:+.2%} (D=1 {net.loc[BASELINE_D, 'cagr']:+.2%}, "
          f"delta {net.loc[champ, 'dCAGR']:+.2%})")
    print("  NOTE: 'highest in history' and 'enough evidence to use' are DIFFERENT questions;")
    print("  the gates below answer the second one, and they answer it one day at a time.")

    tab = gate_table(net, gross, paired, reg, sec, tiers)
    show(tab, ["A_holm", "B_gross", "C_plateau", "D_regimes", "E_decade", "F_riskadj",
               "G_drawdown", "H_tiers", "n_pass", "ALL_EIGHT"],
         "=== the eight gates, per candidate day (D=1 is the baseline, not a candidate) ===")

    winners = tab.index[tab["ALL_EIGHT"]].tolist()
    print(f"\n  gate pass counts: max {int(tab['n_pass'].max())} of 8 "
          f"(day(s) {tab.index[tab['n_pass'] == tab['n_pass'].max()].tolist()})")
    if champ in tab.index:
        row = tab.loc[champ]
        print(f"  the CAGR champion D={champ} passes {int(row['n_pass'])} of 8: "
              + ", ".join(g for g in tab.columns[:8] if bool(row[g])) or "  none")

    print(f"\n  VERDICT: {'CONFIRMED' if winners else 'REJECTED / NOT IDENTIFIED'}")
    if winners:
        print(f"  Day(s) passing all eight pre-registered gates: {winners}")
        print("  Even so, DEPLOYED is NOT modified: the only permitted next step is a separate")
        print("  forward-test registration on genuinely future data (issue #21).")
    else:
        print("  No calendar day passes all eight pre-registered gates. There is no evidence that")
        print("  a fixed calendar day beats the current rule; the historical champion is a")
        print("  post-hoc extremum over 31 enumerations.")

    print(f"\n  Bootstrap winner concentration: max share {wins.max():.1%} "
          f"across {(wins > 0).sum()} distinct winners -> "
          f"{'concentrated' if wins.max() > 0.5 else 'WINNER INSTABILITY'}")
    print("\n  DEPLOYED is unchanged by this study regardless of outcome (issue #21).")
    print("=" * 78)


if __name__ == "__main__":
    main()
