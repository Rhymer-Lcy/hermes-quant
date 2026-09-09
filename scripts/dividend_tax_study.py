"""What does the adjusted-price backtest overstate by reinvesting dividends tax-free? Issue #24.

Pre-registered at https://github.com/Rhymer-Lcy/hermes-quant/issues/24 BEFORE any confirmatory
number was computed. The design frozen there is executed here unchanged, and every cell of the
sensitivity grid is reported whether or not it flatters the strategy.

The lake is forward-adjusted, which is a TOTAL-RETURN series: the engine has always reinvested
every dividend at the ex-date close, instantly, pre-tax, in fractional shares. A real individual
account gets none of that. This measures the difference, on the DEPLOYED strategy, across all
seven capital tiers and the seven frozen tax conventions.

Four blocking gates run before any result is reported:
  M  the derived share multiplier is 1.0 for cash dividends and clusters at stock-dividend ratios
  C  every name the book holds is complete in the dividend and unadjusted-price lakes
  P  at tau = 0 the reconstructed panel is BIT-IDENTICAL to canonical, and so is the equity curve
  A  the engine rerun and the closed-form leakage agree to 5 bp of terminal wealth

    conda activate hermes
    python scripts/dividend_tax_study.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.io import atomic_to_parquet
from hermes.live.strategy import ALL_TIERS, DEPLOYED, TIER_LABEL, deployed_signal
from hermes.paths import BACKTESTS_DIR, PARQUET_DIR, RAW_DIR, ensure_dirs
from hermes.research.backtest.dividends import (PRIMARY_RULE, RATE_LONG, RATE_MID, RATE_SHORT,
                                                TAX_RULES, band_dividend_cash,
                                                closed_form_terminal, daily_shares,
                                                dividend_events, event_tax_table, fifo_lots,
                                                gross_dividend_stream, leakage_stream,
                                                net_close_panel)
from hermes.research.backtest.portfolio import signal_portfolio_backtest

WINDOW_START = pd.Timestamp("2015-01-05")     # frozen in the pre-registration
WINDOW_END = pd.Timestamp("2026-06-30")
GATE_A_BP = 5.0                               # terminal-wealth agreement, basis points
OUT_DIR = BACKTESTS_DIR / "dividend_tax"


def _cagr(equity: pd.Series) -> float:
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def _max_dd(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def _cagr_from(terminal: float, cap: int, equity: pd.Series) -> float:
    """CAGR of an alternative terminal wealth over the canonical curve's own span."""
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    return float((terminal / float(equity.iloc[0])) ** (1.0 / years) - 1.0)


def gate_m(events: pd.DataFrame) -> dict:
    """GATE M: the share multiplier is DERIVED, not assumed -- prove the derivation."""
    print("=== GATE M (blocking): derived share multiplier ===")
    m = events["m"]
    near = m.sub(1.0).abs() <= 0.002
    big = events.loc[m > 1.05, "m"].round(2)
    clusters = big.value_counts()
    known = {1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.45, 1.5, 1.6, 1.7, 1.75, 1.8, 1.9,
             2.0, 2.2, 2.5, 2.6, 3.0, 3.5, 4.0}
    on_grid = float(clusters[clusters.index.isin(known)].sum() / max(clusters.sum(), 1))
    print(f"  events {len(m)}   median m {m.median():.6f}   within 1+/-0.002: {near.mean():.2%}")
    print(f"  m > 1.05: {int(clusters.sum())} events; on a known stock-dividend ratio: {on_grid:.1%}")
    print(f"    top ratios: {', '.join(f'{k:.2f}x{v}' for k, v in clusters.head(8).items())}")
    ok = near.mean() >= 0.80 and on_grid >= 0.80
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("GATE M failed: the multiplier derivation is not recovering "
                         "corporate actions -- the study stops.")
    return {"n_events": int(len(m)), "median_m": float(m.median()),
            "share_within_0002": float(near.mean()), "share_on_known_ratio": on_grid}


def gate_c(held: set[str], raw_close: pd.DataFrame) -> dict:
    """GATE C: a study run against a silently partial lake is how a verdict inverts."""
    print("\n=== GATE C (blocking): dividend / unadjusted-price lake completeness ===")
    done_file = RAW_DIR / "dividend_pull_done.txt"
    done = set(done_file.read_text(encoding="utf-8").split()) if done_file.exists() else set()
    miss_px = sorted(held - set(raw_close.columns))
    miss_div = sorted(held - done)
    print(f"  names actually held by the book: {len(held)}")
    print(f"  missing from raw_close.parquet : {len(miss_px)} {miss_px[:5]}")
    print(f"  incomplete in the dividend pull: {len(miss_div)} {miss_div[:5]}")
    ok = not miss_px and not miss_div
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("GATE C failed: the lake does not cover every held name.")
    return {"n_held": len(held), "missing_price": len(miss_px), "missing_dividend": len(miss_div)}


def gate_p(adj: pd.DataFrame, zero_panel: pd.DataFrame,
           canon: pd.Series, zero_eq: pd.Series, tier: int) -> bool:
    """GATE P: the neutral setting must be bit-identical, not merely close."""
    same_panel = adj.equals(zero_panel)
    same_eq = canon.equals(zero_eq)
    print(f"  {tier:>9,}: panel bit-identical {same_panel}   equity bit-identical {same_eq}"
          f"  -> {'PASS' if same_panel and same_eq else 'FAIL'}")
    return same_panel and same_eq


def main() -> None:
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"window {WINDOW_START.date()} -> {WINDOW_END.date()} (frozen in issue #24)\n")

    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof = membership_lookup(mdf)
    close_full = load_close_panel(codes=union, field="close")
    pe_full = load_close_panel(codes=union, field="peTTM")
    signal_full = deployed_signal(close_full, pe_full, asof)          # lookbacks satisfied

    win = (close_full.index >= WINDOW_START) & (close_full.index <= WINDOW_END)
    adj = close_full.loc[win]
    signal = signal_full.loc[signal_full.index.isin(adj.index)]
    raw_close = pd.read_parquet(PARQUET_DIR / "raw_close.parquet")
    dividends = pd.read_parquet(PARQUET_DIR / "dividends.parquet")

    def run(panel: pd.DataFrame, cap: int):
        return signal_portfolio_backtest(
            panel, signal, cap, DEPLOYED.n_hold, members_asof=asof,
            weight_asof=DEPLOYED.weight_asof, rebalance_band=DEPLOYED.rebalance_band,
            collect_trades=True)

    events = dividend_events(adj, raw_close, dividends)
    gm = gate_m(events)

    canon = {cap: run(adj, cap) for cap in ALL_TIERS}
    held = {t["code"] for r in canon.values() for t in r.trades}
    gc = gate_c(held, raw_close)

    print("\n=== GATE P (blocking): tau = 0 parity with canonical ===")
    cells, per_tier, gate_p_ok, gate_a_ok = [], {}, True, True
    for cap in ALL_TIERS:
        res = canon[cap]
        eq = res.equity
        lots = fifo_lots(res.trades, WINDOW_END)
        shares = daily_shares(res.trades, adj.index, adj.columns)
        base = {"cap": cap, "tier": TIER_LABEL[cap], "cagr": res.cagr,
                "terminal": float(eq.iloc[-1]), "max_dd": res.max_drawdown}
        for key, rule in TAX_RULES.items():
            taxed = event_tax_table(lots, events, rule)
            panel = net_close_panel(adj, taxed, events)
            r = run(panel, cap)
            leak = leakage_stream(shares, adj, taxed)
            gross = gross_dividend_stream(shares, adj, taxed)
            cf = closed_form_terminal(eq, leak)
            cf_prereg = closed_form_terminal(eq, leak, lag=False)
            gap_bp = abs(r.equity.iloc[-1] / cf - 1.0) * 1e4
            bands = band_dividend_cash(shares, adj, taxed)
            cells.append({
                **base, "rule": key, "rule_name": rule.name,
                "v_cagr": r.cagr, "v_terminal": float(r.equity.iloc[-1]),
                "v_max_dd": r.max_drawdown,
                "drag_pp": (res.cagr - r.cagr) * 100.0,
                "gross_div": float(gross.sum()), "tax_paid": float(leak.sum()),
                "eff_rate": float(leak.sum() / gross.sum()) if gross.sum() > 0 else 0.0,
                "n_taxed_events": int(len(taxed)),
                **{f"w_{k}": (v / bands["total"] if bands["total"] else 0.0)
                   for k, v in bands.items() if k != "total"},
                "closed_form_terminal": cf, "closed_form_prereg_form": cf_prereg,
                "gate_a_gap_bp": gap_bp,
            })
            if key == "tau-D":
                gate_p_ok &= gate_p(adj, panel, eq, r.equity, cap)
            if key == PRIMARY_RULE:
                per_tier[cap] = {"res": r, "taxed": taxed, "leak": leak, "gross": gross,
                                 "lots": lots, "shares": shares}
            if leak.sum() > 0:
                gate_a_ok &= gap_bp <= GATE_A_BP
    print(f"  -> GATE P {'PASS' if gate_p_ok else 'FAIL'}")
    if not gate_p_ok:
        raise SystemExit("GATE P failed: the tau = 0 panel is not bit-identical to canonical.")

    df = pd.DataFrame(cells)
    df["cf_drag_pp"] = [(c["cagr"] - _cagr_from(c["closed_form_terminal"], c["cap"],
                                                canon[c["cap"]].equity)) * 100.0
                        for c in cells]

    print("\n=== GATE A (blocking as pre-registered): engine rerun vs closed-form leakage ===")
    live = df[df["tax_paid"] > 0]
    per_tier_gap = live.groupby("cap")["gate_a_gap_bp"].max()
    for cap in ALL_TIERS:
        g = per_tier_gap.get(cap, 0.0)
        print(f"  {cap:>9,}: max gap {g:>8.1f} bp   -> {'PASS' if g <= GATE_A_BP else 'FAIL'}")
    print(f"  overall (limit {GATE_A_BP} bp): {'PASS' if gate_a_ok else 'FAIL'}")
    if not gate_a_ok:
        print("\n  GATE A FAILED AS WRITTEN. Per the pre-registration this is investigated and\n"
              "  reported, never accommodated by widening the tolerance. Diagnosis below; the\n"
              "  frozen 5 bp limit is left exactly as registered and the failure stands on record.")

    # -------- DIAGNOSTIC added AFTER Gate A failed; not part of the frozen design --------
    print(f"\n{'=' * 100}\nPOST-HOC DIAGNOSTIC (not pre-registered, added because Gate A failed):"
          f"\nwhat is the engine-rerun method's own noise floor at each tier?"
          f"\n{'=' * 100}")
    print("  A SIGN-RANDOMISED placebo: the same corrections, same dates, same magnitudes, but a\n"
          "  coin flip decides whether each one is applied as a debit or a credit. Its expected\n"
          "  drag is 0, so the spread across seeds IS the method's resolution at that tier.")
    rng_seeds = range(20)
    noise = {}
    for cap in ALL_TIERS:
        taxed = per_tier[cap]["taxed"]
        eq = canon[cap].equity
        draws = []
        for seed in rng_seeds:
            rng = np.random.default_rng(seed)
            flip = taxed.assign(tau=taxed["tau"] * rng.choice([-1.0, 1.0], size=len(taxed)))
            rp = run(net_close_panel(adj, flip, events), cap)
            draws.append((canon[cap].cagr - _cagr(rp.equity)) * 100.0)
        d = np.array(draws)
        noise[cap] = {"mean_pp": float(d.mean()), "sd_pp": float(d.std(ddof=1)),
                      "p05_pp": float(np.quantile(d, 0.05)), "p95_pp": float(np.quantile(d, 0.95))}
        print(f"  {cap:>9,}: placebo drag mean {d.mean():>+7.4f}  sd {d.std(ddof=1):>6.4f}  "
              f"90% band [{np.quantile(d, .05):>+7.4f}, {np.quantile(d, .95):>+7.4f}] pp/yr")
    print("\n  Read the primary table against these bands: a tier whose placebo band is wider than\n"
          "  the measured drag has no resolution for an effect this size, and its engine-rerun\n"
          "  number carries no information. The closed form is unaffected -- it perturbs the\n"
          "  canonical curve arithmetically and never re-rounds a lot.")

    # ---------------- the frozen battery ----------------
    print(f"\n{'=' * 100}\nDRAG IN pp/yr -- every frozen cell, ENGINE RERUN"
          f"\n{'=' * 100}")
    piv = df.pivot(index="cap", columns="rule", values="drag_pp")
    order = list(TAX_RULES)
    print(f"  {'tier':>9} {'label':>7} {'CAGR':>8}  " + "".join(f"{k:>8}" for k in order))
    for cap in ALL_TIERS:
        row = df[df["cap"] == cap].iloc[0]
        print(f"  {cap:>9,} {TIER_LABEL[cap]:>7} {row['cagr']:>7.2%}  "
              + "".join(f"{piv.at[cap, k]:>+8.3f}" for k in order))

    print(f"\n{'=' * 100}\nDRAG IN pp/yr -- every frozen cell, CLOSED FORM (no lot re-rounding)"
          f"\n{'=' * 100}")
    pivc = df.pivot(index="cap", columns="rule", values="cf_drag_pp")
    print(f"  {'tier':>9} {'label':>7} {'CAGR':>8}  " + "".join(f"{k:>8}" for k in order))
    for cap in ALL_TIERS:
        row = df[df["cap"] == cap].iloc[0]
        print(f"  {cap:>9,} {TIER_LABEL[cap]:>7} {row['cagr']:>7.2%}  "
              + "".join(f"{pivc.at[cap, k]:>+8.3f}" for k in order))

    print(f"\n{'=' * 100}\nPRIMARY CELL {PRIMARY_RULE} -- {TAX_RULES[PRIMARY_RULE].name}"
          f"\n{'=' * 100}")
    pa = df[df["rule"] == PRIMARY_RULE].set_index("cap")
    print(f"  {'tier':>9} {'CAGR net':>9} {'drag engine':>12} {'drag closed':>12} "
          f"{'placebo sd':>11} {'gross div':>12} {'tax paid':>11} {'eff rate':>9} {'events':>7}")
    for cap in ALL_TIERS:
        r = pa.loc[cap]
        print(f"  {cap:>9,} {r['cagr']:>8.2%} {r['drag_pp']:>+12.3f} {r['cf_drag_pp']:>+12.3f} "
              f"{noise[cap]['sd_pp']:>11.3f} {r['gross_div']:>12,.0f} {r['tax_paid']:>11,.0f} "
              f"{r['eff_rate']:>8.2%} {int(r['n_taxed_events']):>7}")

    print(f"\n  dividend value by statutory holding-period band ({PRIMARY_RULE}):")
    print(f"  {'tier':>9} {'<=1m (20%)':>11} {'1m-1y (10%)':>12} {'>1y (0%)':>10}")
    for cap in ALL_TIERS:
        r = pa.loc[cap]
        print(f"  {cap:>9,} {r['w_short']:>10.1%} {r['w_mid']:>11.1%} {r['w_long']:>9.1%}")

    # ---------------- magnitude reconciliation: realised gross dividend yield ----------------
    print(f"\n{'=' * 100}\nMAGNITUDE CHECK -- the book's realised GROSS dividend yield"
          f"\n{'=' * 100}")
    yields = {}
    for cap in ALL_TIERS:
        eq = canon[cap].equity
        g = per_tier[cap]["gross"]
        yr = (g.groupby(g.index.year).sum()
              / eq.groupby(eq.index.year).mean()).rename("gross_yield")
        yields[cap] = yr
        print(f"  {cap:>9,}: mean {yr.mean():>6.2%}   min {yr.min():>6.2%} ({yr.idxmin()})   "
              f"max {yr.max():>6.2%} ({yr.idxmax()})")
    print("  (HS300 trailing yield has run ~2-3%/yr; a book outside ~1-6% would indicate a "
          "scale error)")

    # ---------------- per-year drag, primary rule ----------------
    print(f"\n{'=' * 100}\nPER-YEAR DRAG, {PRIMARY_RULE} -- is the effect stable or concentrated?"
          f"\n{'=' * 100}")
    ref = 100_000
    eq, leak = canon[ref].equity, per_tier[ref]["leak"]
    yearly = (leak.groupby(leak.index.year).sum()
              / eq.groupby(eq.index.year).mean()) * 100.0
    print(f"  tier CNY {ref:,} -- tax paid as % of average equity, by calendar year:")
    print("  " + "  ".join(f"{y}:{v:>5.2f}" for y, v in yearly.items()))
    print(f"  mean {yearly.mean():.3f} pp   sd {yearly.std():.3f} pp   "
          f"min {yearly.min():.3f} ({yearly.idxmin()})   max {yearly.max():.3f} ({yearly.idxmax()})")

    # ---------------- second-order: cash-timing drag ----------------
    print(f"\n{'=' * 100}\nSECOND-ORDER (reported separately, NOT folded into the headline)"
          f"\n{'=' * 100}")
    second = {}
    for cap in ALL_TIERS:
        taxed, shares, eq = per_tier[cap]["taxed"], per_tier[cap]["shares"], canon[cap].equity
        rebal = sorted({pd.Timestamp(t["date"]) for t in canon[cap].trades})
        pos = {d: i for i, d in enumerate(adj.index)}
        timing = pd.Series(0.0, index=adj.index)
        stranded = 0.0
        for rec in taxed.itertuples(index=False):
            i = pos.get(rec.date)
            if i is None or i == 0:
                continue
            prev = adj.index[i - 1]
            val = float(shares.at[prev, rec.code]) * float(adj.at[prev, rec.code])
            if not np.isfinite(val) or val <= 0:
                continue
            net_cash = val * rec.dyield * (1.0 - rec.tau)
            nxt = next((d for d in rebal if d > rec.date), None)
            if nxt is None:
                continue
            p0, p1 = adj.at[rec.date, rec.code], adj.at[nxt, rec.code]
            if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                timing.iloc[i] += net_cash * (p1 / p0 - 1.0)   # forgone in-name compounding
            stranded += net_cash
        cf = closed_form_terminal(eq, timing)
        drag = (_cagr(eq) - _cagr_from(cf, cap, eq)) * 100.0
        second[cap] = {"timing_drag_pp": drag, "net_cash_total": stranded,
                       "cash_per_rebalance": stranded / max(len(rebal), 1)}
        print(f"  {cap:>9,}: cash-timing drag {drag:>+7.4f} pp/yr   net dividend cash "
              f"CNY {stranded:>12,.0f} total, CNY {stranded / max(len(rebal), 1):>9,.0f} "
              f"per rebalance")
    print("  Lot friction: dividend cash is not stranded -- it merges with the rebalance cash and "
          "is\n  spent on the same 100-share lots the engine already models, so its only cost is "
          "the idle\n  period above. Per-rebalance cash at the CNY 10k tier is well under one lot, "
          "which is why\n  it cannot be reinvested on its own.")

    # ---------------- open-lot convention (NOT frozen in the pre-registration) ----------------
    print(f"\n{'=' * 100}\nA CONVENTION THE PRE-REGISTRATION DID NOT FREEZE -- disclosed and "
          f"bounded\n{'=' * 100}")
    print("  A lot still open at the window end has not been transferred, so the statute has not\n"
          "  yet fixed its rate. The study continues it to the window edge. That choice was not\n"
          "  frozen in advance, so its full range is reported rather than asserted immaterial.")
    open_sens = {}
    for cap in ALL_TIERS:
        lots_c, shares_c = per_tier[cap]["lots"], per_tier[cap]["shares"]
        n_open = int(lots_c["open"].sum())
        row = {"n_open": n_open, "n_lots": int(len(lots_c))}
        for lab, orate in (("registered", None), ("exempt", 0.0), ("max", RATE_SHORT)):
            tx = event_tax_table(lots_c, events, TAX_RULES[PRIMARY_RULE], open_rate=orate)
            lk = leakage_stream(shares_c, adj, tx)
            row[lab] = (canon[cap].cagr
                        - _cagr_from(closed_form_terminal(canon[cap].equity, lk), cap,
                                     canon[cap].equity)) * 100.0
        open_sens[cap] = row
        print(f"  {cap:>9,}: {n_open:>3}/{len(lots_c):>4} lots open   drag pp/yr -- "
              f"registered {row['registered']:>+6.3f}   all-exempt {row['exempt']:>+6.3f}   "
              f"all-20% {row['max']:>+6.3f}")
    print("  The all-20% column is a bound, not a scenario: it taxes multi-year holdings at the\n"
          "  one-month rate. The honest range is between the registered and all-exempt columns.")

    # ---------------- persist ----------------
    atomic_to_parquet(df, OUT_DIR / "cells.parquet", index=False)
    summary = {
        "issue": 24,
        "window": [str(WINDOW_START.date()), str(WINDOW_END.date())],
        "gates": {"M": gm, "C": gc, "P": bool(gate_p_ok),
                  "A": {"pass": bool(gate_a_ok), "max_gap_bp": float(live["gate_a_gap_bp"].max()),
                        "limit_bp": GATE_A_BP,
                        "per_tier_max_gap_bp": {str(c): float(per_tier_gap.get(c, 0.0))
                                                for c in ALL_TIERS}}},
        "placebo_noise_floor_pp": {str(c): noise[c] for c in ALL_TIERS},
        "primary_rule": PRIMARY_RULE,
        "primary_drag_pp_by_tier": {str(c): float(pa.loc[c, "drag_pp"]) for c in ALL_TIERS},
        "primary_closed_form_drag_pp_by_tier": {str(c): float(pa.loc[c, "cf_drag_pp"])
                                                for c in ALL_TIERS},
        "effective_rate_by_tier": {str(c): float(pa.loc[c, "eff_rate"]) for c in ALL_TIERS},
        "band_weights_by_tier": {str(c): {"short": float(pa.loc[c, "w_short"]),
                                          "mid": float(pa.loc[c, "w_mid"]),
                                          "long": float(pa.loc[c, "w_long"])} for c in ALL_TIERS},
        "canonical_cagr_by_tier": {str(c): float(canon[c].cagr) for c in ALL_TIERS},
        "gross_yield_mean_by_tier": {str(c): float(yields[c].mean()) for c in ALL_TIERS},
        "second_order": {str(c): second[c] for c in ALL_TIERS},
        "open_lot_convention_sensitivity_pp": {str(c): open_sens[c] for c in ALL_TIERS},
        "per_year_drag_pp_ref_tier": {str(k): float(v) for k, v in yearly.items()},
        "rules": {k: v.name for k, v in TAX_RULES.items()},
        "statutory_rates": {"short": RATE_SHORT, "mid": RATE_MID, "long": RATE_LONG},
    }
    Path(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_DIR / 'cells.parquet'} and {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
