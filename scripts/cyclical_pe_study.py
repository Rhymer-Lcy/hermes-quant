"""Inverse-PE timing on cyclicals: buy high PE, sell low PE. Issue #4.

The friend treats the tech and infrastructure sleeves as cyclicals and inverts the value rule
there -- start building when PE is HIGH (earnings trough; a negative PE counts as the deepest
trough), sell when profits are high (PE compressed). Pre-registered before this script existed:

  set        PIT HS300+CSI500 members, non-ST, >= 20 prior traded days, in the tech or
             infrastructure buckets of the frozen CSRC mapping (issue #4 appendix)
  enter      peTTM < 0, or peTTM >= its own trailing 1200d 70th percentile over positive-PE
             observations (>= 750 total observations required for the gate to exist)
  exit       0 < peTTM <= its own trailing 30th percentile over positive-PE observations
  arms       P_inv (the rule), P_bh (hold all cyclicals), P_mirror (thresholds swapped)
  portfolio  equal weight, monthly rebalance, signal month-end t -> execute t+1 close,
             proportional retail costs
  verdict    CONFIRMED only if P_inv beats P_bh net with monthly-clustered t > 2 AND beats
             P_mirror on point estimate

    conda activate hermes
    python scripts/cyclical_pe_study.py
"""
from __future__ import annotations

import pandas as pd

from hermes.data.fundamentals import CYCLICAL_BUCKETS, INDUSTRY_PARQUET
from hermes.data.lake import load_close_panel
from hermes.data.membership import (CSI500_MEMBERSHIP_PARQUET, MEMBERSHIP_PARQUET,
                                    membership_lookup)
from hermes.io import atomic_to_parquet
from hermes.paths import BACKTESTS_DIR, ensure_dirs
from hermes.research.backtest.metrics import clustered_tstat, sharpe
from hermes.research.backtest.rule_portfolio import monthly_ew_backtest, state_mask

PCT_HI, PCT_LO = 0.70, 0.30
WINDOW, MIN_OBS = 1200, 750
MIN_HISTORY = 20
ERAS = [("2015-01-01", "2019-12-31"), ("2020-01-01", "2026-12-31")]


def pe_gates(pe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(high, low, valid): the frozen entry/exit conditions. `high` is peTTM negative or at or
    above the name's own trailing 70th percentile of positive-PE observations; `low` is peTTM
    positive and at or below the 30th; `valid` requires 750 total trailing observations."""
    valid = pe.rolling(WINDOW, min_periods=1).count() >= MIN_OBS
    pe_pos = pe.where(pe > 0)
    q_hi = pe_pos.rolling(WINDOW, min_periods=1).quantile(PCT_HI)
    q_lo = pe_pos.rolling(WINDOW, min_periods=1).quantile(PCT_LO)
    high = valid & pe.notna() & ((pe < 0) | (pe >= q_hi))
    low = valid & (pe > 0) & (pe <= q_lo)
    return high, low, valid


def _line(tag: str, strat: pd.Series, bench: pd.Series) -> dict:
    active = (strat - bench).dropna()
    return {"tag": tag,
            "wealth_strat": float((1 + strat.dropna()).prod()),
            "wealth_bench": float((1 + bench.dropna()).prod()),
            "ann_active": float(active.mean() * 252),
            "sharpe_strat": sharpe(strat), "sharpe_bench": sharpe(bench),
            "t_month": clustered_tstat(active, active.index, freq="M")}


def _print(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'variant':>26} {'wealth':>8} {'bench':>8} {'active/yr':>10} {'Sharpe':>7} "
          f"{'bench':>7} {'t(month)':>9}")
    for r in rows:
        print(f"  {r['tag']:>26} {r['wealth_strat']:>8.3f} {r['wealth_bench']:>8.3f} "
              f"{r['ann_active']:>+10.2%} {r['sharpe_strat']:>7.2f} {r['sharpe_bench']:>7.2f} "
              f"{r['t_month']:>9.2f}")


def main() -> None:
    ensure_dirs()
    ind = pd.read_parquet(INDUSTRY_PARQUET)
    buckets = ind.set_index("code")["bucket"]
    hs = pd.read_parquet(MEMBERSHIP_PARQUET)
    cs = pd.read_parquet(CSI500_MEMBERSHIP_PARQUET)
    union = sorted(set(hs["code"]) | set(cs["code"]))
    cyc = [c for c in union if buckets.get(c) in CYCLICAL_BUCKETS]
    print(f"cyclical set: {len(cyc)} of {len(union)} PIT union names "
          f"({', '.join(CYCLICAL_BUCKETS)} buckets)")

    close = load_close_panel(codes=cyc, field="close")
    pe = load_close_panel(codes=cyc, field="peTTM")
    st = load_close_panel(codes=cyc, field="isST")
    hs_asof, cs_asof = membership_lookup(hs), membership_lookup(cs)
    member = pd.DataFrame({c: [c in hs_asof(d) or c in cs_asof(d) for d in close.index]
                           for c in close.columns}, index=close.index)
    base = (member & ~st.eq(True) & close.notna()
            & (close.notna().cumsum().shift(1) >= MIN_HISTORY))
    ret = close.pct_change(fill_method=None)

    high, low, valid = pe_gates(pe)
    arms = {
        "P_inv": state_mask(high, low) & base,       # the friend's rule
        "P_bh": base,                                # hold all cyclicals
        "P_mirror": state_mask(low, high) & base,    # the swapped direction
    }
    res = {k: monthly_ew_backtest(m, ret) for k, m in arms.items()}
    inv, bh, mirror = res["P_inv"]["net"], res["P_bh"]["net"], res["P_mirror"]["net"]
    print(f"gate coverage: valid on {valid.mean().mean():.1%} of name-days; held share "
          f"P_inv {arms['P_inv'].sum(axis=1).mean():.0f} vs bh "
          f"{arms['P_bh'].sum(axis=1).mean():.0f} names/day")

    rows = [_line("P_inv vs hold-all NET", inv, bh),
            _line("P_mirror vs hold-all NET", mirror, bh),
            _line("P_inv vs P_mirror", inv, mirror)]
    for s, e in ERAS:
        sl = slice(pd.Timestamp(s), pd.Timestamp(e))
        rows.append(_line(f"P_inv vs hold {s[:4]}-{e[:4]}", inv.loc[sl], bh.loc[sl]))
    for bucket in CYCLICAL_BUCKETS:
        names = [c for c in cyc if buckets.get(c) == bucket]
        sub = {k: monthly_ew_backtest(m[names], ret[names]) for k, m in arms.items()
               if k != "P_mirror"}
        rows.append(_line(f"{bucket} only", sub["P_inv"]["net"], sub["P_bh"]["net"]))
    _print(rows, "INVERSE-PE ON CYCLICALS (buy high PE, sell low PE):")

    head = rows[0]
    beats_bh = head["ann_active"] > 0 and head["t_month"] > 2
    beats_mirror = float((inv - mirror).dropna().mean()) > 0
    verdict = beats_bh and beats_mirror
    print(f"\nVERDICT: {'CONFIRMED' if verdict else 'REJECTED'} -- needs P_inv > hold-all with "
          f"monthly t > 2 (got {head['t_month']:.2f}) AND P_inv > P_mirror on point estimate "
          f"({'yes' if beats_mirror else 'no'})")

    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "cyclical_pe_summary.parquet",
                      index=False)
    atomic_to_parquet(pd.DataFrame({"inv_net": inv, "hold_all": bh, "mirror_net": mirror}),
                      BACKTESTS_DIR / "cyclical_pe_daily.parquet")
    print(f"saved -> {BACKTESTS_DIR / 'cyclical_pe_summary.parquet'}")


if __name__ == "__main__":
    main()
