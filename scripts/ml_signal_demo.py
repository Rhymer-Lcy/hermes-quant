"""Walk-forward LightGBM multi-factor combiner -- the survivorship-free, no-look-ahead
test of whether combining factors beats a single factor on HS300.

Pipeline: 5 factors (earnings/book yield, low-vol, momentum, reversal) -> standardized
features -> walk-forward LightGBM (train only on the past) -> out-of-sample signal ->
evaluate OOS IC/quantile -> backtest through the PIT friction engine. Outputs go to
results/ (signals, backtests).

    python scripts/ml_signal_demo.py
"""
import pandas as pd

from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.paths import BACKTESTS_DIR, SIGNALS_DIR, ensure_dirs
from hermes.research.backtest.portfolio import signal_portfolio_backtest
from hermes.research.eval.factor_eval import compute_ic, quantile_returns
from hermes.research.factors import library as fl
from hermes.research.model.walk_forward import build_dataset, walk_forward_predict

TIERS = [100_000, 1_000_000]
N_HOLD = 10


def main() -> None:
    ensure_dirs()
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof = membership_lookup(mdf)
    close = load_close_panel(codes=union, field="close")
    pe = load_close_panel(codes=union, field="peTTM")
    pb = load_close_panel(codes=union, field="pbMRQ")
    eval_dates = [pd.Timestamp(d) for d in sorted(mdf["date"].unique())
                  if pd.Timestamp(d) in close.index]

    factors = {
        "ep": fl.earnings_yield(pe),
        "bp": fl.book_yield(pb),
        "low_vol": fl.low_vol(close, 120),
        "mom": fl.momentum(close, 120, 20),
        "rev": -fl.trailing_return(close, 20),
    }

    print("building dataset + walk-forward LightGBM (train only on the past)...")
    data, cols = build_dataset(factors, close, eval_dates, members_asof=asof)
    signal = walk_forward_predict(data, cols, min_train=24, window=36)
    out = SIGNALS_DIR / "ml_5factor.parquet"
    signal.to_parquet(out)
    pred_dates = list(signal.index)
    print(f"OOS signal: {len(pred_dates)} months "
          f"({pred_dates[0].date()}..{pred_dates[-1].date()}), saved -> {out}\n")

    # Out-of-sample evaluation of the combined signal.
    ic = compute_ic(signal, close, pred_dates, members_asof=asof)
    q = quantile_returns(signal, close, pred_dates, n_q=5, members_asof=asof)
    spread = (q.iloc[-1] - q.iloc[0]) if len(q) == 5 else float("nan")
    print(f"ML combined signal  OOS: mean IC {ic.mean_ic:+.4f}  IC IR {ic.ic_ir:+.3f}  "
          f"t-stat {ic.t_stat:+.2f}  hit {ic.hit_rate*100:.0f}%  Q5-Q1 {spread:+.2%}/mo\n")

    # Backtest the ML signal vs single factors uniformly (all via signal_portfolio_backtest,
    # PIT + frictions) -- the FAIR benchmark is the best single factor, not just momentum.
    strategies = {
        "ML-5factor": signal,
        "low_vol": fl.low_vol(close, 120),
        "ep (1/PE)": fl.earnings_yield(pe),
        "momentum": close / close.shift(20) - 1.0,
    }
    print(f"  {'capital':>12} {'strategy':>11} {'held':>6} {'net total':>11} {'CAGR':>8} {'maxDD':>8}")
    rows = []
    for cap in TIERS:
        for tag, sig in strategies.items():
            r = signal_portfolio_backtest(close, sig, cap, N_HOLD, members_asof=asof)
            print(f"  {cap:>12,} {tag:>11} {r.avg_names_held:>6.1f} {r.total_return:>+11.1%} "
                  f"{r.cagr:>+8.1%} {r.max_drawdown:>8.1%}")
            rows.append({"capital": cap, "strategy": tag, "held": r.avg_names_held,
                         "total_return": r.total_return, "cagr": r.cagr, "max_dd": r.max_drawdown})
    pd.DataFrame(rows).to_csv(BACKTESTS_DIR / "ml_vs_factors_pit.csv", index=False)

    print("\nAll survivorship-free (PIT membership) + no-look-ahead (walk-forward training). "
          "Honest read: the ML OOS IC (+0.024) is BELOW low-vol alone (+0.067), so stacking weak "
          "factors (momentum/reversal) DILUTED the signal. Compare the ML backtest to low_vol / ep "
          "single factors -- beating momentum is a low bar.")


if __name__ == "__main__":
    main()
