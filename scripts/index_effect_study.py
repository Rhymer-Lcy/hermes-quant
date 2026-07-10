"""HS300 index-reconstitution effect: is the US delete-reversal edge also present in A-shares?

MOTIVATION. The sibling plutus-quant validated exactly one retail-operable satellite edge in US
equities: S&P 500 DELETIONS earn a +3.4% to +5.1% abnormal bounce net of costs over 20-60 days
(docs/index_effect_study.md there). The mechanism -- forced selling by index trackers pushes the
dropped name below fair value, then it mean-reverts -- is not US-specific, and A-share index
tracking AUM has grown sharply. hermes already holds everything needed: PIT month-end HS300
membership and daily bars for every name ever in the index (the survivorship-free union).

DESIGN, and what the monthly snapshots honestly measure. HS300 reconstitution is semiannual
(effective around mid-June / mid-December); membership here is month-END snapshots, so a change is
DETECTED at the first snapshot after it becomes effective -- typically 2-3 weeks late. Events are
therefore anchored at the detection snapshot, entering at the NEXT trading day's close (the same
T+1 convention as the engine). This is precisely the version a retail replicator of this repo could
trade, and it deliberately misses the first ~2-3 post-effective weeks; if the US-style bounce is
front-loaded, the lag UNDERSTATES the edge, so a positive read here is conservative.

Abnormal return = the name's daily return minus the PIT-universe equal-weight mean that day (the
plutus convention). A name whose bars end inside the horizon (delisting -- deletions include names
dropped BECAUSE they are dying) has its cumulative abnormal return frozen at its last bar: the
position is carried at its final real price, matching the engine's forced-exit bookkeeping, and the
terminal collapse it suffered up to that point stays in the average -- dead names are NOT excluded.

INFERENCE. Same-batch events share almost all of their return window, so single events are not
independent draws. The t-statistic is computed across BATCH means (one mean abnormal return per
reconstitution batch), the honest independence unit (~28 batches).

    python scripts/index_effect_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.data.ingest import BACKTEST_END
from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.paths import BACKTESTS_DIR
from hermes.research.backtest.frictions import AShareCosts

HORIZONS = [5, 10, 20, 40, 60]
ROUND_TRIP = AShareCosts().slip * 2 + 2.5e-4 * 2 + 5e-4 + 1e-5 * 2   # ~20.2 bps, matches frictions.py


def detect_events(mdf: pd.DataFrame) -> pd.DataFrame:
    """(snapshot, code, kind) rows: 'del' = in previous snapshot but not this one, 'add' = reverse."""
    snaps = sorted(mdf["date"].unique())
    sets = {d: set(mdf.loc[mdf["date"] == d, "code"]) for d in snaps}
    rows = []
    for prev, curr in zip(snaps, snaps[1:]):
        for c in sorted(sets[prev] - sets[curr]):
            rows.append({"snapshot": pd.Timestamp(curr), "code": c, "kind": "del"})
        for c in sorted(sets[curr] - sets[prev]):
            rows.append({"snapshot": pd.Timestamp(curr), "code": c, "kind": "add"})
    return pd.DataFrame(rows)


def universe_mean_returns(ret: pd.DataFrame, asof) -> pd.Series:
    """Per-date equal-weight mean daily return over the PIT universe AS OF that date. Membership is
    piecewise-constant between monthly snapshots, so each date is one masked row-mean."""
    out = pd.Series(index=ret.index, dtype=float)
    for d in ret.index:
        members = asof(d)
        if not members:
            out.loc[d] = 0.0
            continue
        row = ret.loc[d]
        row = row[row.index.isin(members)].dropna()
        out.loc[d] = float(row.mean()) if len(row) else 0.0
    return out


def event_caar(events: pd.DataFrame, close: pd.DataFrame, asof) -> pd.DataFrame:
    """Per event: cumulative abnormal return at each horizon, from entry (first trading day after
    the detection snapshot). Frozen at the last real bar if the name's series ends (delisting)."""
    ret = close.pct_change(fill_method=None)   # a halted day is NO TRADE (NaN), not a zero return
    univ = universe_mean_returns(ret, asof)
    dates = close.index
    rows = []
    for ev in events.itertuples():
        pos = dates.searchsorted(ev.snapshot, side="right")   # entry bar = first day after snapshot
        if pos + 1 >= len(dates):
            continue                                          # no forward window at the lake edge
        window = slice(pos + 1, pos + 1 + max(HORIZONS))      # returns AFTER the entry close
        ab = (ret[ev.code].iloc[window] - univ.iloc[window]).to_numpy()
        if len(ab) == 0:
            continue
        # freeze after the last real bar: a dead day contributes log1p(0)=0, so the cumulative
        # value (including any terminal collapse already suffered) carries forward unchanged
        cum = np.cumsum(np.log1p(np.where(np.isnan(ab), 0.0, ab)))
        out = {"snapshot": ev.snapshot, "code": ev.code, "kind": ev.kind,
               "days_alive": int((~np.isnan(ab)).sum())}
        for h in HORIZONS:
            out[f"d{h}"] = float(np.expm1(cum[min(h, len(cum)) - 1]))
        rows.append(out)
    return pd.DataFrame(rows)


def batch_t(df: pd.DataFrame, col: str) -> tuple[float, float, int]:
    """Mean and t-stat across BATCH means (one value per reconstitution snapshot)."""
    b = df.groupby("snapshot")[col].mean().dropna()
    n = len(b)
    if n < 2:
        return float("nan"), float("nan"), n
    return float(b.mean()), float(b.mean() / (b.std(ddof=1) / np.sqrt(n))), n


def main() -> None:
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    mdf = mdf[mdf["date"] <= pd.Timestamp(BACKTEST_END)]      # pinned window, as everywhere
    union = sorted(mdf["code"].unique())
    asof = membership_lookup(mdf)
    close = load_close_panel(codes=union, field="close")      # UNPINNED on purpose: a late-window
    # event needs its 60 forward days, which lie after BACKTEST_END; only events are pinned.

    events = detect_events(mdf)
    print(f"HS300 reconstitution events through {BACKTEST_END}: "
          f"{(events['kind'] == 'del').sum()} deletions, {(events['kind'] == 'add').sum()} additions, "
          f"{events['snapshot'].nunique()} batches")
    print("NOTE: detection is at the month-end snapshot AFTER the effective date (~2-3 weeks late)"
          " -- this measures the retail-implementable, lagged entry, not the announcement effect.\n")

    caar = event_caar(events, close, asof)
    caar.to_csv(BACKTESTS_DIR / "index_effect_events.csv", index=False)

    for kind, label in (("del", "DELETE (reversal?)"), ("add", "ADD")):
        sub = caar[caar["kind"] == kind]
        dead = int((sub["days_alive"] < max(HORIZONS)).sum())
        print(f"  {label}: n={len(sub)} events ({dead} die inside the 60d window and stay in)")
        print(f"    {'day':>6} {'mean CAAR':>10} {'batch-t':>8} {'batches':>8} {'hit%':>6}")
        for h in HORIZONS:
            m, t, nb = batch_t(sub, f"d{h}")
            hit = float((sub[f"d{h}"] > 0).mean()) * 100
            print(f"    {h:>6} {m:>+10.2%} {t:>+8.2f} {nb:>8} {hit:>5.0f}%")
        print()

    m60, t60, _ = batch_t(caar[caar["kind"] == "del"], "d60")
    print(f"net read: DELETE mean CAAR d60 {m60:+.2%} minus ~{ROUND_TRIP:.2%} round-trip = "
          f"{m60 - ROUND_TRIP:+.2%} net (t={t60:+.2f} across batches)")
    print("\nReading: the US edge is a POSITIVE delete CAAR that clears the round-trip cost. If the"
          " A-share read is ~0 or negative, the lagged monthly detection may have missed a"
          " front-loaded bounce, or A-share deletions (often demotions into CSI500 with heavy"
          " retail ownership) simply do not carry the forced-selling dislocation -- either way,"
          " REPORT the number; do not tune the window to manufacture one.")


if __name__ == "__main__":
    main()
