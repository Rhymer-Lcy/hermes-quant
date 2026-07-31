"""Is the bad-preannouncement edge worth anything long-only, as an avoidance overlay? Issue #20.

Study #19 established that a negative earnings preannouncement is followed by significant negative
drift (-2.39% / 60d, t = -3.45; -4.18% / 120d, t = -5.08) while positive ones do not drift. That
edge is on the short side. The one long-only expression is AVOIDANCE: refuse to hold a name that
recently preannounced badly.

An event study cannot price this. The value of an exclusion is (loss avoided) minus (opportunity
cost of the replacement that takes the slot) -- a fixed-size book always holds ten names, so
skipping one means buying the eleventh. Only a portfolio backtest running the real selection, the
real replacement and the real frictions can settle it. Design frozen in issue #20 BEFORE any
portfolio number was computed:

  base       the DEPLOYED spec, unchanged; the overlay is the ONLY difference
  exclusion  negative-type preannouncement (预减/略减/首亏/续亏/增亏) with 公告日期 in (T-90d, T]
  mechanism  excluded -> NaN score; the engine drops NaN before ranking, slot passes to the next
             name, book size stays 10
  PIT        only announcements with 公告日期 <= T are visible at T
  tiers      all seven live.strategy.ALL_TIERS

PRIMARY: net Calmar improves at the 1M tier AND the sign is identical across all seven tiers.
INCONCLUSIVE if the overlay changes fewer than 20 name-months: "the rule never fired" and "the
rule fired and did not help" are different findings. The exclusion count prints FIRST, before any
performance number, so power decides the branch rather than outcome.
FALSIFICATION: the mirror book (hold ONLY the excluded names) must underperform. If it does not,
the primary is noise whatever its sign.

    python scripts/preannounce_avoid_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.data.ingest import BACKTEST_END
from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.live.strategy import ALL_TIERS, DEPLOYED, deployed_signal
from hermes.paths import BACKTESTS_DIR, PARQUET_DIR
from hermes.research.backtest.frictions import AShareCosts
from hermes.research.backtest.portfolio import signal_portfolio_backtest

LAKE = PARQUET_DIR / "preannounce.parquet"
CSI500_MEMBERSHIP = PARQUET_DIR / "csi500_membership.parquet"
REFERENCE_TIER = 1_000_000
MIN_NAME_MONTHS = 20                      # frozen power floor -> INCONCLUSIVE below this

NEGATIVE = {"预减", "略减", "首亏", "续亏", "增亏"}
SEVERE = {"首亏", "续亏", "增亏"}          # an outright loss, not merely a smaller profit
UNCERTAIN = {"不确定"}
WINDOWS = [60, 90, 180]                   # calendar days; 90 is the primary
PRIMARY_WINDOW = 90


# --- events -> exclusion mask --------------------------------------------------------

def to_bs_code(code: str) -> str:
    c = str(code).zfill(6)
    return ("sh." if c[0] in "69" else "sz.") + c


def negative_events(kinds: set[str]) -> pd.DataFrame:
    """Deduplicated preannouncements of the requested kinds, exactly as #19 deduplicated them:
    earliest 公告日期 per (name, period), 净利润 preferred on ties, revisions excluded."""
    df = pd.read_parquet(LAKE).rename(
        columns={"股票代码": "code_raw", "公告日期": "ann", "预告类型": "kind",
                 "预测指标": "metric", "报告期": "period", "序号": "seq"})
    df["ann"] = pd.to_datetime(df["ann"], errors="coerce")
    df = df.dropna(subset=["ann", "code_raw"])
    df["code"] = df["code_raw"].map(to_bs_code)
    df["metric_rank"] = (df["metric"].astype(str) != "净利润").astype(int)
    df = df.sort_values(["code", "period", "ann", "metric_rank", "seq"])
    first = df.groupby(["code", "period"], as_index=False).first()
    return first[first["kind"].isin(kinds)][["code", "ann", "kind"]]


def exclusion_mask(index: pd.DatetimeIndex, columns: pd.Index,
                   events: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Boolean panel: True where a name is inside (ann, ann + window] of a qualifying event.

    Strictly after the announcement date, matching #19's entry convention -- an announcement
    released during or after a session must not affect that session."""
    mask = pd.DataFrame(False, index=index, columns=columns)
    col_pos = {c: i for i, c in enumerate(columns)}
    vals = mask.to_numpy()
    for ev in events.itertuples():
        j = col_pos.get(ev.code)
        if j is None:
            continue
        lo = index.searchsorted(ev.ann, side="right")             # strictly after
        hi = index.searchsorted(ev.ann + pd.Timedelta(days=window_days), side="right")
        if hi > lo:
            vals[lo:hi, j] = True
    return pd.DataFrame(vals, index=index, columns=columns)


# --- metrics -------------------------------------------------------------------------

def calmar(r) -> float:
    return r.cagr / abs(r.max_drawdown) if r.max_drawdown else float("nan")


#: every fee and the slippage zeroed -- cell (6) checks that any apparent gain is not a cost artefact
ZERO_COST = AShareCosts(commission_rate=0.0, min_commission=0.0, stamp_tax_sell=0.0,
                        transfer_fee_rate=0.0, slippage_bps=0.0)


def run(close, signal, cap, n_hold, asof, costs: bool = True):
    return signal_portfolio_backtest(close, signal, cap, n_hold, members_asof=asof,
                                     rebalance_band=DEPLOYED.rebalance_band,
                                     costs=None if costs else ZERO_COST)


def name_months_changed(signal: pd.DataFrame, mask: pd.DataFrame, asof, n_hold: int) -> int:
    """How many (rebalance month, name) pairs the overlay actually changed.

    The operationally honest power metric: not how many names were masked, but how many times the
    mask altered the book that would otherwise have been held."""
    changed = 0
    for sd in signal.resample("ME").last().index:
        if sd not in signal.index:
            prior = signal.index[signal.index <= sd]
            if len(prior) == 0:
                continue
            sd = prior[-1]
        members = asof(sd)
        row = signal.loc[sd].dropna()
        row = row[[c for c in row.index if c in members]]
        if row.empty:
            continue
        base = set(row.sort_values(ascending=False).index[:n_hold])
        kept = row[~mask.loc[sd, row.index].to_numpy()]
        over = set(kept.sort_values(ascending=False).index[:n_hold])
        changed += len(base - over)
    return changed


def fmt_row(label: str, r) -> str:
    return (f"  {label:<34} CAGR {r.cagr:>+7.1%}  maxDD {r.max_drawdown:>7.1%}  "
            f"Calmar {calmar(r):>5.2f}  held {r.avg_names_held:>4.1f}")


def main() -> None:
    # --- panels -------------------------------------------------------------------
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    close = load_close_panel(codes=union, end=BACKTEST_END)
    pe = load_close_panel(codes=union, field="peTTM", end=BACKTEST_END)
    asof = membership_lookup(mdf)
    sig = deployed_signal(close, pe, asof, DEPLOYED)

    ev_neg = negative_events(NEGATIVE)
    mask = exclusion_mask(close.index, close.columns, ev_neg, PRIMARY_WINDOW)

    # --- POWER FIRST (frozen: decides the INCONCLUSIVE branch before any performance read) ---
    print("=== POWER (reported before any performance number, per the pre-registration) ===")
    print(f"  negative-type events in the lake      : {len(ev_neg):,}")
    print(f"  events touching the HS300 price panel : "
          f"{ev_neg['code'].isin(close.columns).sum():,}")
    changed = name_months_changed(sig, mask, asof, DEPLOYED.n_hold)
    print(f"  name-months the overlay CHANGED       : {changed}")
    print(f"  frozen floor                          : {MIN_NAME_MONTHS}")
    underpowered = changed < MIN_NAME_MONTHS
    print(f"  -> {'INCONCLUSIVE (underpowered)' if underpowered else 'powered; proceed'}")

    # --- PRIMARY: tier sweep ------------------------------------------------------
    print(f"\n=== PRIMARY: deployed vs overlay, {PRIMARY_WINDOW}d window, all tiers ===")
    sig_over = sig.where(~mask)
    rows = []
    for cap in ALL_TIERS:
        b = run(close, sig, cap, DEPLOYED.n_hold, asof)
        o = run(close, sig_over, cap, DEPLOYED.n_hold, asof)
        rows.append({"cap": cap, "base_cagr": b.cagr, "over_cagr": o.cagr,
                     "base_dd": b.max_drawdown, "over_dd": o.max_drawdown,
                     "base_calmar": calmar(b), "over_calmar": calmar(o),
                     "d_calmar": calmar(o) - calmar(b)})
    t = pd.DataFrame(rows)
    for _, r in t.iterrows():
        print(f"  {r['cap']:>9,.0f}  base {r['base_cagr']:>+6.1%}/{r['base_dd']:>6.1%}/"
              f"{r['base_calmar']:>5.2f}   overlay {r['over_cagr']:>+6.1%}/{r['over_dd']:>6.1%}/"
              f"{r['over_calmar']:>5.2f}   dCalmar {r['d_calmar']:>+5.2f}")

    ref = t[t["cap"] == REFERENCE_TIER].iloc[0]
    signs = set(np.sign(t["d_calmar"].round(4)))
    consistent = len(signs - {0.0}) <= 1 and (t["d_calmar"] > 0).all()
    if underpowered:
        verdict = "INCONCLUSIVE (underpowered)"
    elif ref["d_calmar"] > 0 and consistent:
        verdict = "CONFIRMED"
    else:
        verdict = "REJECTED"
    print("\n  frozen rule: net Calmar improves at 1M AND the sign is identical across all tiers")
    print(f"  read: dCalmar@1M {ref['d_calmar']:+.2f}, all-tier improvement {bool((t['d_calmar'] > 0).all())}"
          f"  ->  **{verdict}**")

    # --- FALSIFICATION: the mirror ------------------------------------------------
    print("\n=== FALSIFICATION: the mirror book (hold ONLY the excluded names) ===")
    print("  if avoidance helps, the mirror must underperform; if it does not, the primary is noise")
    sig_mirror = sig.where(mask)
    b = run(close, sig, REFERENCE_TIER, DEPLOYED.n_hold, asof)
    m = run(close, sig_mirror, REFERENCE_TIER, DEPLOYED.n_hold, asof)
    print(fmt_row("deployed (reference)", b))
    print(fmt_row("mirror: bad-news names only", m))
    print(f"  mirror underperforms: {m.cagr < b.cagr}")

    # --- ROBUSTNESS GRID (frozen; reference tier unless stated) -------------------
    print(f"\n=== ROBUSTNESS GRID (frozen in #20; {REFERENCE_TIER:,} tier) ===")

    print("\n-- (1) window --")
    for w in WINDOWS:
        mk = exclusion_mask(close.index, close.columns, ev_neg, w)
        r = run(close, sig.where(~mk), REFERENCE_TIER, DEPLOYED.n_hold, asof)
        tag = " [primary]" if w == PRIMARY_WINDOW else ""
        print(fmt_row(f"{w}d window{tag}", r))
    print(fmt_row("no overlay (baseline)", b))

    print("\n-- (2) exclusion set --")
    for label, kinds in [("negative [primary]", NEGATIVE), ("negative + 不确定", NEGATIVE | UNCERTAIN),
                         ("severe only (首亏/续亏/增亏)", SEVERE)]:
        mk = exclusion_mask(close.index, close.columns, negative_events(kinds), PRIMARY_WINDOW)
        print(fmt_row(label, run(close, sig.where(~mk), REFERENCE_TIER, DEPLOYED.n_hold, asof)))

    print("\n-- (5) era split (overlay minus baseline, within era) --")
    for lo, hi, name in [("2015-01-01", "2019-12-31", "2015-2019"),
                         ("2020-01-01", "2026-12-31", "2020-2026")]:
        c2 = close.loc[lo:hi]
        s2 = sig.loc[lo:hi]
        m2 = mask.loc[lo:hi]
        rb = run(c2, s2, REFERENCE_TIER, DEPLOYED.n_hold, asof)
        ro = run(c2, s2.where(~m2), REFERENCE_TIER, DEPLOYED.n_hold, asof)
        print(f"  {name}: base Calmar {calmar(rb):>5.2f}  overlay {calmar(ro):>5.2f}  "
              f"dCalmar {calmar(ro) - calmar(rb):>+5.2f}")

    print("\n-- (6) cost sensitivity: zero-cost gross --")
    print(fmt_row("baseline, zero cost", run(close, sig, REFERENCE_TIER, DEPLOYED.n_hold, asof, costs=False)))
    print(fmt_row("overlay, zero cost", run(close, sig.where(~mask), REFERENCE_TIER, DEPLOYED.n_hold, asof, costs=False)))

    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
    t.to_csv(BACKTESTS_DIR / "preannounce_avoid_tiers.csv", index=False)
    print(f"\nsaved -> {BACKTESTS_DIR / 'preannounce_avoid_tiers.csv'}")


if __name__ == "__main__":
    main()
