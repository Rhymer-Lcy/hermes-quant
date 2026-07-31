"""Does the mandatory earnings preannouncement (业绩预告) pay, long-only? Issue #19.

China mandates a preannouncement whenever a result crosses disclosure thresholds -- a scheduled,
cross-sectionally comparable disclosure event with no direct analogue in mature markets. Design
frozen in the issue BEFORE this script existed and before any event data was pulled:

  data       akshare stock_yjyg_em per reporting period 2015-2025 (data/parquet/preannounce.parquet)
  PIT anchor 公告日期 -- the vendor's `date=` is the PERIOD, so a FY2024 row can announce in 2025;
             the panel is assembled by announcement date and nothing enters before it is announced
  dedup      one event per (name, period): earliest 公告日期, ties prefer 预测指标 == 净利润, then
             the vendor's 序号. LATER announcements for the same period are revisions -> excluded
             from the primary, reported separately
  universe   PIT HS300 + CSI500 members, non-ST, >= 20 prior traded days, events 2015+
  signal     cross-sectional RANK of 业绩变动幅度 within the announcement month (rank, not raw:
             a percentage change off a near-zero base is unbounded -- frozen as an outlier
             decision, not a tunable knob)
  entry      close of the first trading day STRICTLY AFTER 公告日期
  horizons   20 / 60 / 120 trading days, abnormal vs the equal-weight universe mean
  costs      0.20% round trip charged AGAINST the long leg

PRIMARY VERDICT: the top-quintile LONG LEG's 60d net abnormal mean POSITIVE with monthly-clustered
t > 2. The Q5-Q1 spread is secondary (monotonicity evidence) and can never alone confirm -- study
#16 found a significant index-addition effect that was unharvestable because collecting it needed
shorting, and this design refuses to repeat that.

CO-PRIMARY: correlation of a monthly top-quintile portfolio with the deployed strategy. The
deployed book's -33% drawdown survived A1-A9; what it lacks is an UNCORRELATED signal, so a
confirmed-but-correlated signal is operationally useless. Deployment-worthy = correlation < 0.5
AND blended Calmar > the deployed 0.32.

    python scripts/preannounce_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.data.ingest import BACKTEST_END
from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.paths import BACKTESTS_DIR, PARQUET_DIR

LAKE = PARQUET_DIR / "preannounce.parquet"
CSI500_MEMBERSHIP = PARQUET_DIR / "csi500_membership.parquet"
HORIZONS = [20, 60, 120]
ROUND_TRIP = 0.0020
N_BUCKETS_PRIMARY = 5
MIN_PRIOR_BARS = 20
POSITIVE_TYPES = {"预增", "略增", "扭亏", "续盈"}
NEGATIVE_TYPES = {"预减", "略减", "首亏", "续亏"}


# --- event construction -------------------------------------------------------------

def to_bs_code(code: str) -> str:
    """'600406' -> 'sh.600406'. 6xx/9xx are Shanghai; the rest Shenzhen."""
    c = str(code).zfill(6)
    return ("sh." if c[0] in "69" else "sz.") + c


def load_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicated preannouncements, one per (code, period), with the revision flag."""
    df = pd.read_parquet(LAKE)
    df = df.rename(columns={"股票代码": "code_raw", "公告日期": "ann", "业绩变动幅度": "pct",
                            "预告类型": "kind", "预测指标": "metric", "报告期": "period",
                            "序号": "seq"})
    df["ann"] = pd.to_datetime(df["ann"], errors="coerce")
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
    df = df.dropna(subset=["ann", "code_raw"])
    df["code"] = df["code_raw"].map(to_bs_code)
    # disclosure lag: announcement date minus period end. Feeds robustness cell (6), the one
    # angle that is not a PEAD restatement -- earliness is orthogonal to the news content.
    df["lag_days"] = (df["ann"] - pd.to_datetime(df["period"], format="%Y%m%d")).dt.days

    # earliest announcement per (code, period); ties -> 净利润 first, then the vendor's 序号
    df["metric_rank"] = (df["metric"].astype(str) != "净利润").astype(int)
    df = df.sort_values(["code", "period", "ann", "metric_rank", "seq"])
    first = df.groupby(["code", "period"], as_index=False).first()
    first["is_revision"] = False
    later = df.merge(first[["code", "period", "ann"]], on=["code", "period"],
                     suffixes=("", "_first"))
    later = later[later["ann"] > later["ann_first"]].copy()
    later["is_revision"] = True
    return first, later


# --- universe / returns -------------------------------------------------------------

def pit_universe() -> tuple:
    """PIT HS300 + CSI500 membership: ONE LOOKUP PER INDEX, unioned per query date.

    Deliberately not a concatenated frame. The two indices do not share a snapshot grid (HS300
    carries 6 snapshots CSI500 lacks), and `membership_lookup` groups by date -- so on a date
    present in only one index, a concatenated frame yields a snapshot containing ONLY that
    index's codes and silently drops the other from the universe. Unioning two independent
    as-of lookups gives each index its own latest-snapshot-on-or-before semantics.
    """
    frames = {"hs300": pd.read_parquet(MEMBERSHIP_PARQUET)}
    if CSI500_MEMBERSHIP.exists():
        frames["csi500"] = pd.read_parquet(CSI500_MEMBERSHIP)
    lookups = [membership_lookup(f) for f in frames.values()]
    codes = sorted({c for f in frames.values() for c in f["code"].unique()})

    def asof(when) -> set[str]:
        out: set[str] = set()
        for lk in lookups:
            out |= lk(when)
        return out

    return codes, asof


def forward_abnormal(px: pd.DataFrame, events: pd.DataFrame, horizons: list[int],
                     st: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per event: the entry bar (first session strictly after 公告日期) and, per horizon, the
    stock's forward return minus the equal-weight universe mean over the same bars.

    `st` is the isST panel; the frozen universe is non-ST, judged AT THE ENTRY BAR (the moment
    the rule would actually buy), not at the announcement or with any later information."""
    idx = px.index
    ew = px.pct_change().mean(axis=1)                       # equal-weight universe daily return
    pos_of = {d: i for i, d in enumerate(idx)}
    rows = []
    for ev in events.itertuples():
        code = ev.code
        if code not in px.columns:
            continue
        after = idx[idx > ev.ann]
        if len(after) == 0:
            continue
        entry = after[0]
        i0 = pos_of[entry]
        if i0 < MIN_PRIOR_BARS:                             # >= 20 prior traded bars
            continue
        s = px[code]
        if not np.isfinite(s.iloc[i0]):
            continue
        if st is not None and code in st.columns:           # frozen: non-ST at the entry bar
            flag = st[code].iloc[i0]
            if pd.notna(flag) and bool(flag):
                continue
        rec = {"code": code, "period": ev.period, "ann": ev.ann, "entry": entry,
               "pct": ev.pct, "kind": ev.kind, "lag_days": ev.lag_days}
        for h in horizons:
            i1 = i0 + h
            if i1 >= len(idx) or not np.isfinite(s.iloc[i1]):
                rec[f"ar{h}"] = np.nan
                continue
            stock = s.iloc[i1] / s.iloc[i0] - 1.0
            bench = float((1.0 + ew.iloc[i0 + 1:i1 + 1].fillna(0.0)).prod() - 1.0)
            rec[f"ar{h}"] = stock - bench
        rows.append(rec)
    return pd.DataFrame(rows)


def clustered_t(df: pd.DataFrame, col: str) -> tuple[float, float, int]:
    """Mean and monthly-clustered t. Same-month events share their return window, so the unit of
    inference is the MONTH, not the event (repo convention across the ruleset studies)."""
    d = df.dropna(subset=[col])
    if d.empty:
        return np.nan, np.nan, 0
    monthly = d.groupby(d["entry"].dt.to_period("M"))[col].mean()
    if len(monthly) < 2:
        return float(d[col].mean()), np.nan, len(d)
    t = float(monthly.mean() / (monthly.std(ddof=1) / np.sqrt(len(monthly))))
    return float(d[col].mean()), t, len(d)


# --- reporting ----------------------------------------------------------------------

def bucket_table(ev: pd.DataFrame, n_buckets: int, label: str, rank_col: str = "pct") -> pd.DataFrame:
    """Rank events into buckets WITHIN each announcement month, then report the long leg net of
    the round trip, plus the top-minus-bottom spread (secondary)."""
    d = ev.dropna(subset=[rank_col]).copy()
    d["bucket"] = (d.groupby(d["entry"].dt.to_period("M"))[rank_col]
                    .transform(lambda s: pd.qcut(s.rank(method="first"), n_buckets,
                                                 labels=False, duplicates="drop") + 1
                               if s.notna().sum() >= n_buckets else np.nan))
    d = d.dropna(subset=["bucket"])
    out = []
    for h in HORIZONS:
        top = d[d["bucket"] == n_buckets]
        bot = d[d["bucket"] == 1]
        m_top, t_top, n_top = clustered_t(top, f"ar{h}")
        m_bot, _, _ = clustered_t(bot, f"ar{h}")
        out.append({"label": label, "h": h, "n_top": n_top,
                    "top_gross": m_top, "top_net": m_top - ROUND_TRIP if pd.notna(m_top) else np.nan,
                    "top_t": t_top, "bot_gross": m_bot,
                    "spread": m_top - m_bot if pd.notna(m_top) and pd.notna(m_bot) else np.nan})
    return pd.DataFrame(out)


def fmt(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in ["top_gross", "top_net", "bot_gross", "spread"]:
        if c in d:
            d[c] = d[c].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "--")
    if "top_t" in d:
        d["top_t"] = d["top_t"].map(lambda v: f"{v:+.2f}" if pd.notna(v) else "--")
    return d.to_string(index=False)


def main() -> None:
    first, revisions = load_events()
    union, asof = pit_universe()
    px = load_close_panel(codes=union, end=BACKTEST_END)
    st = load_close_panel(codes=union, field="isST", end=BACKTEST_END)

    # PIT universe membership at the announcement date
    keep = [bool(r.code in asof(r.ann)) for r in first.itertuples()]
    ev_all = first[keep].copy()
    ev_all = ev_all[ev_all["ann"] >= pd.Timestamp("2015-01-01")]

    print("=== descriptive annex (no verdict weight) ===")
    print(f"  raw lake rows           : {len(pd.read_parquet(LAKE)):,}")
    print(f"  deduplicated events     : {len(first):,}   revisions excluded: {len(revisions):,}")
    print(f"  in PIT universe, 2015+  : {len(ev_all):,}")
    print(f"  announcement lag (days) : median {ev_all['lag_days'].median():.0f}, "
          f"p10 {ev_all['lag_days'].quantile(.1):.0f}, p90 {ev_all['lag_days'].quantile(.9):.0f}")
    print("  by 预告类型:")
    for k, n in ev_all["kind"].value_counts().head(12).items():
        print(f"    {k:>6}: {n:,}")

    ev = forward_abnormal(px, ev_all, HORIZONS, st)
    print(f"\n  events with a tradable entry bar: {len(ev):,}")

    print("\n=== PRIMARY: magnitude-rank quintiles, long leg vs EW universe ===")
    primary = bucket_table(ev, N_BUCKETS_PRIMARY, "magnitude rank (Q5)")
    print(fmt(primary))
    row60 = primary[primary["h"] == 60].iloc[0]
    verdict = "CONFIRMED" if (pd.notna(row60["top_t"]) and row60["top_net"] > 0
                              and row60["top_t"] > 2) else "REJECTED"
    print("\n  frozen rule: 60d top-quintile NET > 0 with monthly-clustered t > 2")
    print(f"  read: net {row60['top_net']:+.2%}, t = {row60['top_t']:+.2f}  ->  **{verdict}**")

    print("\n=== ROBUSTNESS GRID (frozen in the issue; every cell reported) ===")
    print("\n-- (2b) categorical: positive vs negative 预告类型 --")
    rows = []
    for name, kinds in [("positive", POSITIVE_TYPES), ("negative", NEGATIVE_TYPES)]:
        sub = ev[ev["kind"].isin(kinds)]
        for h in HORIZONS:
            m, t, n = clustered_t(sub, f"ar{h}")
            rows.append({"set": name, "h": h, "n": n, "gross": m,
                         "net": m - ROUND_TRIP if pd.notna(m) else np.nan, "t": t})
    r = pd.DataFrame(rows)
    for c in ["gross", "net"]:
        r[c] = r[c].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "--")
    r["t"] = r["t"].map(lambda v: f"{v:+.2f}" if pd.notna(v) else "--")
    print(r.to_string(index=False))

    print("\n-- (2c) magnitude rank WITHIN positive types only --")
    print(fmt(bucket_table(ev[ev["kind"].isin(POSITIVE_TYPES)], N_BUCKETS_PRIMARY, "pos-only Q5")))

    print("\n-- (3) terciles --")
    print(fmt(bucket_table(ev, 3, "magnitude rank (T3)")))

    print("\n-- (4) era split --")
    for lo, hi, name in [("2015-01-01", "2019-12-31", "2015-2019"),
                         ("2020-01-01", "2026-12-31", "2020-2026")]:
        sub = ev[(ev["entry"] >= lo) & (ev["entry"] <= hi)]
        print(fmt(bucket_table(sub, N_BUCKETS_PRIMARY, name)))

    print("\n-- (5) annual (12-31) vs interim periods --")
    for name, mask in [("annual", ev["period"].astype(str).str.endswith("1231")),
                       ("interim", ~ev["period"].astype(str).str.endswith("1231"))]:
        print(fmt(bucket_table(ev[mask], N_BUCKETS_PRIMARY, name)))

    print("\n-- (6) announcement earliness terciles (good news early?) --")
    e = ev.dropna(subset=["lag_days"]).copy()
    e["early"] = pd.qcut(e["lag_days"].rank(method="first"), 3, labels=["earliest", "mid", "latest"])
    rows = []
    for name, sub in e.groupby("early", observed=True):
        for h in HORIZONS:
            m, t, n = clustered_t(sub, f"ar{h}")
            rows.append({"tercile": name, "h": h, "n": n, "gross": m,
                         "net": m - ROUND_TRIP if pd.notna(m) else np.nan, "t": t})
    r = pd.DataFrame(rows)
    for c in ["gross", "net"]:
        r[c] = r[c].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "--")
    r["t"] = r["t"].map(lambda v: f"{v:+.2f}" if pd.notna(v) else "--")
    print(r.to_string(index=False))

    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(BACKTESTS_DIR / "preannounce_events.parquet", index=False)
    primary.to_csv(BACKTESTS_DIR / "preannounce_primary.csv", index=False)
    print(f"\nsaved -> {BACKTESTS_DIR}")


if __name__ == "__main__":
    main()
