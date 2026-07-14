"""Liquidity/turnover factor battery on PIT HS300 and CSI500 -- from lake fields turn/amount.

Two purposes, one script:

1. HS300: the one factor family never tested on the deployed universe. factor_ic_study covered
   value/momentum/reversal/low-vol; A2-A4 covered weighting/sector/size. Battery here: low
   turnover, low turnover-volatility, Amihud illiquidity -- orientations fixed a-priori in the
   factor library (docstrings there), not chosen after looking.

2. CSI500: restore the reproducibility of the "CSI500-native factors" follow-up recorded in
   docs/risk_control.md, whose original script was never committed and whose dataset was purged.
   Re-measure on the regenerated, coverage-gated dataset: turnstd20's IC (documented as +0.084,
   t=6.15, hit 74%, positive in every 2y sub-period), and its long-only top-30 read (documented
   as Calmar ~0.20 / maxDD ~-44% / ~0.75 correlation to the deployed HS300 book).

PRE-REGISTERED GATE: a factor proceeds to a long-only backtest only where its monthly rank-IC has
|t| >= 2; the backtest config is fixed in advance (HS300 top-10 plain; CSI500 top-30 with
price-limit no-fill ON and ST filtered -- the small-cap-rigorous treatment), at CNY 1,000,000 with
A-share frictions. No weight or window sweeps.

    python scripts/liquidity_factor_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.data.ingest import BACKTEST_END
from hermes.data.lake import load_close_panel
from hermes.data.membership import (CSI500_MEMBERSHIP_PARQUET, MEMBERSHIP_PARQUET,
                                    membership_lookup)
from hermes.live.strategy import deployed_signal
from hermes.paths import BACKTESTS_DIR
from hermes.research.backtest.limits import limit_flags
from hermes.research.backtest.portfolio import signal_portfolio_backtest
from hermes.research.eval.factor_eval import compute_ic
from hermes.research.factors import library as fl

CAP = 1_000_000
IC_T_GATE = 2.0
SUBPERIODS = [("2015-2016", "2015-01-01", "2016-12-31"), ("2017-2018", "2017-01-01", "2018-12-31"),
              ("2019-2020", "2019-01-01", "2020-12-31"), ("2021-2022", "2021-01-01", "2022-12-31"),
              ("2023-2025", "2023-01-01", BACKTEST_END)]


def load(parquet, fields=("close", "turn", "amount", "isST", "preclose")):
    mdf = pd.read_parquet(parquet)
    union = sorted(mdf["code"].unique())
    asof = membership_lookup(mdf)
    panels = {f: load_close_panel(codes=union, field=f, end=BACKTEST_END) for f in fields}
    have = int(panels["close"].notna().any().sum())
    if have / len(union) < 0.99:                       # same completeness gate as the A6 study
        raise RuntimeError(f"universe data INCOMPLETE for {parquet}: {have}/{len(union)} names "
                           f"({have / len(union):.1%} < 99%). Rebuild with build_csi500_dataset.py.")
    eval_dates = [pd.Timestamp(d) for d in sorted(mdf["date"].unique())
                  if pd.Timestamp(d) in panels["close"].index]
    return union, asof, panels, eval_dates


def battery(panels) -> dict[str, pd.DataFrame]:
    return {
        "low_turn20": fl.low_turnover(panels["turn"], 20),
        "low_turnstd20": fl.low_turnover_vol(panels["turn"], 20),
        "amihud20": fl.amihud_illiquidity(panels["close"], panels["amount"], 20),
    }


def ic_table(factors, close, eval_dates, asof, label: str) -> dict[str, object]:
    print(f"\n[{label}] monthly rank-IC, PIT universe, {len(eval_dates)} eval dates:")
    print(f"  {'factor':>14} {'mean IC':>8} {'IC IR':>7} {'t':>6} {'hit%':>6}")
    out = {}
    for name, f in factors.items():
        r = compute_ic(f, close, eval_dates, members_asof=asof)
        out[name] = r
        print(f"  {name:>14} {r.mean_ic:>+8.4f} {r.ic_ir:>+7.3f} {r.t_stat:>+6.2f} "
              f"{r.hit_rate * 100:>5.0f}%")
    return out


def cal(r) -> float:
    return r.cagr / abs(r.max_drawdown) if r.max_drawdown < 0 else float("nan")


def main() -> None:
    # ---- HS300 -----------------------------------------------------------------------
    _, hs_asof, hs, hs_eval = load(MEMBERSHIP_PARQUET, fields=("close", "turn", "amount", "peTTM"))
    hs_factors = {k: fl.restrict_to_universe(v, hs_asof) for k, v in battery(hs).items()}
    hs_ic = ic_table(hs_factors, hs["close"], hs_eval, hs_asof, "HS300")

    rows = []
    for name, r in hs_ic.items():
        rows.append({"universe": "HS300", "factor": name, "mean_ic": r.mean_ic,
                     "t": r.t_stat, "hit": r.hit_rate})
        if abs(r.t_stat) >= IC_T_GATE:
            bt = signal_portfolio_backtest(hs["close"], hs_factors[name], CAP, 10,
                                           members_asof=hs_asof)
            print(f"  -> {name} passes the IC gate (|t|>={IC_T_GATE:.0f}): top-10 long-only "
                  f"CAGR {bt.cagr:+.1%}  maxDD {bt.max_drawdown:.1%}  Calmar {cal(bt):.2f}")
            rows[-1].update({"lo_cagr": bt.cagr, "lo_maxdd": bt.max_drawdown, "lo_calmar": cal(bt)})

    # ---- CSI500 (native-factors re-verification) --------------------------------------
    _, cs_asof, cs, cs_eval = load(CSI500_MEMBERSHIP_PARQUET)
    # .eq(True), not ~cs["isST"]: the flag panel carries NaN where a name has no bar, and eq()
    # maps those to False (= not ST), which a bitwise ~ on a float panel cannot do.
    not_st = ~cs["isST"].eq(True)                      # ST filtered, as in the A6 study
    cs_factors = {k: fl.restrict_to_universe(v.where(not_st), cs_asof)
                  for k, v in battery(cs).items()}
    cs_ic = ic_table(cs_factors, cs["close"], cs_eval, cs_asof, "CSI500 (ST filtered)")

    print("\n  turnstd20 IC by 2y sub-period (documented claim: positive in every one):")
    ic_series = compute_ic(cs_factors["low_turnstd20"], cs["close"], cs_eval,
                           members_asof=cs_asof).ic
    for lab, lo, hi in SUBPERIODS:
        s = ic_series[(ic_series.index >= lo) & (ic_series.index <= hi)].dropna()
        t = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else float("nan")
        print(f"    {lab}: mean IC {s.mean():+.4f}  t {t:+.2f}  n {len(s)}")

    for name, r in cs_ic.items():
        rows.append({"universe": "CSI500", "factor": name, "mean_ic": r.mean_ic,
                     "t": r.t_stat, "hit": r.hit_rate})

    lb = limit_flags(cs["close"], cs["preclose"])      # price-limit no-fill ON for small caps
    bt = signal_portfolio_backtest(cs["close"], cs_factors["low_turnstd20"], CAP, 30,
                                   members_asof=cs_asof, limit_block=lb)
    dep = signal_portfolio_backtest(hs["close"],
                                    deployed_signal(hs["close"], hs["peTTM"], hs_asof),
                                    CAP, 10, members_asof=hs_asof)
    corr = (bt.equity.resample("ME").last().pct_change()
            .corr(dep.equity.resample("ME").last().pct_change()))
    print(f"\n  turnstd20 CSI500 top-30 long-only (limits ON, ST filtered): "
          f"CAGR {bt.cagr:+.1%}  maxDD {bt.max_drawdown:.1%}  Calmar {cal(bt):.2f}  "
          f"monthly corr to deployed HS300 book {corr:+.2f}")
    rows.append({"universe": "CSI500", "factor": "low_turnstd20 top-30 LO",
                 "lo_cagr": bt.cagr, "lo_maxdd": bt.max_drawdown, "lo_calmar": cal(bt),
                 "corr_vs_deployed": corr})

    pd.DataFrame(rows).to_csv(BACKTESTS_DIR / "liquidity_factors.csv", index=False)
    print(f"\nwrote liquidity_factors.csv to {BACKTESTS_DIR}")
    print("Reading: HS300 -- a liquidity factor earns a backtest only past the |t|>=2 gate, and "
          "earns ADOPTION only by beating the deployed book's Calmar 0.32, which no single factor "
          "here is expected to do (this is a completeness test of the last untested family). "
          "CSI500 -- this re-measures the native-factors follow-up on the regenerated dataset; "
          "its verdict (real left-tail signal, unharvestable long-only) stands or falls on these "
          "numbers, and docs/risk_control.md is updated to whatever they say.")


if __name__ == "__main__":
    main()
