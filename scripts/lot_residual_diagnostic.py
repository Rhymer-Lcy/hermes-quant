"""BASELINE-ONLY diagnostic of integer-lot residual cash. No candidate allocator here.

This characterises what the CURRENT production allocator does, historically and strictly before
the paper inception, so that a study of alternatives can be motivated by measurement rather than
by one quarter's observation. Nothing in this file compares allocators.

The production allocator floors each name's target INDEPENDENTLY:

    shares_i = floor( (gross / n_hold) / (p_i * (1 + slip) * lot) ) * lot

which creates residual cash through two mechanically different channels:

  ZERO-SLICE   the name's equal-weight slice cannot buy even one lot
               (gross/n_hold < lot * p_i * (1+slip)) -- the whole slice stays cash
  REMAINDER    the slice buys k lots with a leftover smaller than one more lot

The two behave completely differently in capital: REMAINDER shrinks like 1/capital, ZERO-SLICE is
a step function of the basket's price level. Separating them is the point of this file.

    conda activate hermes
    python scripts/lot_residual_diagnostic.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.live.strategy import ALL_TIERS, DEPLOYED, PAPER_INCEPTION, deployed_signal
from hermes.paths import RESULTS_DIR, ensure_dirs
from hermes.research.backtest.frictions import AShareCosts
from hermes.research.backtest.portfolio import signal_portfolio_backtest

INCEPTION = pd.Timestamp(PAPER_INCEPTION)
START = pd.Timestamp("2015-01-05")
OUT = RESULTS_DIR / "reviews"
COSTS = AShareCosts()
LOT, SLIP = COSTS.lot_size, COSTS.slip
N_HOLD = DEPLOYED.n_hold


def replay_states(res, close: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Post-rebalance state at every rebalance: cash ratio, names held, basket price level.

    Reconstructed from the engine's own fill log and equity curve, so it is the real book and
    not a re-simulation."""
    tr = pd.DataFrame(res.trades)
    if tr.empty:
        return pd.DataFrame()
    tr["date"] = pd.to_datetime(tr["date"])
    eq = res.equity
    pos: dict[str, int] = {}
    rows = []
    for d in sorted(tr["date"].unique()):
        day = tr[tr["date"] == d]
        for _i, t in day.iterrows():
            pos[t["code"]] = pos.get(t["code"], 0) + int(t["shares"])
        held = {c: s for c, s in pos.items() if s > 0}
        px = close.loc[d]
        mv = float(sum(s * px.get(c, np.nan) for c, s in held.items()))
        equity = float(eq.loc[d])
        if not np.isfinite(mv) or equity <= 0:
            continue
        prices = [float(px.get(c, np.nan)) for c in held]
        prices = [p for p in prices if np.isfinite(p)]
        rows.append({
            "date": d, "cap": cap, "equity": equity,
            "cash": equity - mv, "cash_ratio": (equity - mv) / equity,
            "n_held": len(held),
            "basket_median_px": float(np.median(prices)) if prices else np.nan,
            "basket_max_px": float(np.max(prices)) if prices else np.nan,
            "slice": equity / N_HOLD,
        })
    return pd.DataFrame(rows)


def channel_split(res, close: pd.DataFrame, signal: pd.DataFrame, asof, cap: int) -> pd.DataFrame:
    """Split residual cash into ZERO-SLICE and REMAINDER at each rebalance, using the SAME
    target rule the engine applies, evaluated on the engine's own pre-trade equity."""
    tr = pd.DataFrame(res.trades)
    tr["date"] = pd.to_datetime(tr["date"])
    eq = res.equity
    dates = close.index
    pos_of = {d: i for i, d in enumerate(dates)}
    rows = []
    for d in sorted(tr["date"].unique()):
        i = pos_of.get(pd.Timestamp(d))
        if i is None or i == 0:
            continue
        sd = dates[i - 1]                                  # the signal bar the engine read
        if sd not in signal.index:
            continue
        f = signal.loc[sd].dropna()
        raw = close.iloc[i]
        f = f[raw.reindex(f.index).notna()]
        f = f[f.index.isin(asof(sd))]
        top = f.sort_values(ascending=False).index.tolist()[:N_HOLD]
        if not top:
            continue
        equity = float(eq.iloc[i - 1])                     # pre-trade equity proxy (prior close)
        slice_val = equity / N_HOLD
        zero, rem = 0.0, 0.0
        for c in top:
            p = float(raw[c]) * (1 + SLIP)
            lot_cost = p * LOT
            k = int(slice_val // lot_cost)
            if k == 0:
                zero += slice_val
            else:
                rem += slice_val - k * lot_cost
        rows.append({"date": d, "cap": cap, "equity": equity,
                     "zero_slice": zero, "remainder": rem,
                     "zero_pct": zero / equity, "rem_pct": rem / equity,
                     "n_affordable": sum(1 for c in top
                                         if slice_val >= float(raw[c]) * (1 + SLIP) * LOT)})
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof = membership_lookup(mdf)
    close_full = load_close_panel(codes=union, field="close")
    pe_full = load_close_panel(codes=union, field="peTTM")
    sig_full = deployed_signal(close_full, pe_full, asof)

    hist = close_full.loc[(close_full.index >= START) & (close_full.index < INCEPTION)]
    sig = sig_full.loc[sig_full.index.isin(hist.index)]
    print(f"BASELINE ONLY. Historical window {hist.index[0].date()} -> {hist.index[-1].date()} "
          f"({len(hist)} bars), strictly before the {INCEPTION.date()} paper inception.\n")

    states, splits = {}, {}
    for cap in ALL_TIERS:
        res = signal_portfolio_backtest(hist, sig, cap, N_HOLD, members_asof=asof,
                                        weight_asof=DEPLOYED.weight_asof,
                                        rebalance_band=DEPLOYED.rebalance_band,
                                        collect_trades=True)
        states[cap] = replay_states(res, hist, cap)
        splits[cap] = channel_split(res, hist, sig, asof, cap)

    # ---------------- capital curves (B10) ----------------
    print("=" * 104)
    print("CAPITAL CURVES -- post-rebalance state across all historical rebalances")
    print("=" * 104)
    print(f"  {'tier':>9} {'rebal':>6} {'cash med':>9} {'cash p90':>9} {'names med':>10} "
          f"{'all-10 %':>9} {'names p10':>10} {'zero-slice med':>15} {'remainder med':>14}")
    curve = []
    for cap in ALL_TIERS:
        s, sp = states[cap], splits[cap]
        row = {
            "cap": cap, "n_rebal": int(len(s)),
            "cash_med": float(s["cash_ratio"].median()),
            "cash_p90": float(s["cash_ratio"].quantile(0.90)),
            "names_med": float(s["n_held"].median()),
            "all10_share": float((s["n_held"] >= N_HOLD).mean()),
            "names_p10": float(s["n_held"].quantile(0.10)),
            "zero_med": float(sp["zero_pct"].median()),
            "rem_med": float(sp["rem_pct"].median()),
        }
        curve.append(row)
        print(f"  {cap:>9,} {row['n_rebal']:>6} {row['cash_med']:>9.2%} {row['cash_p90']:>9.2%} "
              f"{row['names_med']:>10.1f} {row['all10_share']:>9.1%} {row['names_p10']:>10.1f} "
              f"{row['zero_med']:>15.2%} {row['rem_med']:>14.2%}")

    print("\n  Reading: REMAINDER falls roughly like 1/capital (pure lot rounding, irreducible "
          "for a\n  given basket). ZERO-SLICE is a step function of the basket's price level and "
          "is where an\n  independent floor throws away a whole position.")

    # ---------------- the ratchet claim (B9) ----------------
    print("\n" + "=" * 104)
    print("IS THE CASH RATIO A RATCHET? -- consecutive-rebalance changes, historical")
    print("=" * 104)
    print(f"  {'tier':>9} {'n diffs':>8} {'up %':>7} {'down %':>7} {'mean d':>9} "
          f"{'lag-1 autocorr of level':>24} {'longest monotone up-run':>24}")
    ratchet = []
    for cap in ALL_TIERS:
        s = states[cap].sort_values("date")
        c = s["cash_ratio"].to_numpy()
        d = np.diff(c)
        run = best = 0
        for x in d:
            run = run + 1 if x > 0 else 0
            best = max(best, run)
        ac = float(pd.Series(c).autocorr(lag=1)) if len(c) > 3 else np.nan
        ratchet.append({"cap": cap, "up": float((d > 0).mean()), "mean_d": float(d.mean()),
                        "autocorr": ac, "longest_up_run": int(best), "n": int(len(d))})
        print(f"  {cap:>9,} {len(d):>8} {(d > 0).mean():>7.1%} {(d < 0).mean():>7.1%} "
              f"{d.mean():>+9.4%} {ac:>24.3f} {best:>24}")

    # ---------------- price-level dependence ----------------
    print("\n" + "=" * 104)
    print("WHAT DRIVES IT? -- cash ratio vs basket price level (per tier, Spearman)")
    print("=" * 104)
    print(f"  {'tier':>9} {'rho(cash, median basket px)':>28} {'rho(cash, slice/lotcost)':>26}")
    for cap in ALL_TIERS:
        s = states[cap].dropna(subset=["basket_median_px"])
        if len(s) < 20:
            continue
        r1 = float(s["cash_ratio"].corr(s["basket_median_px"], method="spearman"))
        afford = s["slice"] / (s["basket_median_px"] * LOT)
        r2 = float(s["cash_ratio"].corr(afford, method="spearman"))
        print(f"  {cap:>9,} {r1:>28.3f} {r2:>26.3f}")
    print("\n  'slice/lotcost' is how many lots one equal-weight slice can buy at the basket's "
          "median price.\n  A strong NEGATIVE rank correlation there is the signature of the "
          "zero-slice channel.")

    # ---------------- compare with the documented threshold ----------------
    print("\n" + "=" * 104)
    print("AGAINST THE EXISTING DOCUMENTED THRESHOLD (docs/paper_trading.md, strategy.py)")
    print("=" * 104)
    print("  documented: '>= CNY 30k to start'; 'at CNY 10k the book is nearly full (9.6/10)';")
    print("              CNY 10k INFEASIBLE, CNY 30k the floor, CNY 50k comfortably viable.")
    for cap in (10_000, 30_000, 50_000, 100_000):
        s = states[cap]
        print(f"  measured {cap:>8,}: mean names held {s['n_held'].mean():.2f}, "
              f"median {s['n_held'].median():.1f}, all-10 on {float((s['n_held'] >= N_HOLD).mean()):.1%} "
              f"of rebalances, median cash {s['cash_ratio'].median():.1%}")

    pd.concat(states.values()).to_parquet(OUT / "lot_states_baseline.parquet", index=False)
    pd.concat(splits.values()).to_parquet(OUT / "lot_channels_baseline.parquet", index=False)
    (OUT / "lot_diagnostic_baseline.json").write_text(
        json.dumps({"window": [str(hist.index[0].date()), str(hist.index[-1].date())],
                    "curve": curve, "ratchet": ratchet}, indent=2, default=float),
        encoding="utf-8")
    print(f"\nwrote {OUT / 'lot_diagnostic_baseline.json'}")


if __name__ == "__main__":
    main()
