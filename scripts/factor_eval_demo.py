"""Single-factor IC + quantile evaluation on the point-in-time HS300 universe.

The rigorous gate before any ML: which raw factors actually predict next-month
returns out-of-sample (survivorship-free)? Rank IC is computed on non-overlapping
month-end dates over the then-current HS300 members.

    python scripts/factor_eval_demo.py
"""
import pandas as pd

from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.research.eval.factor_eval import compute_ic, quantile_returns
from hermes.research.factors import library as fl


def main() -> None:
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof = membership_lookup(mdf)

    close = load_close_panel(codes=union, field="close")
    pe = load_close_panel(codes=union, field="peTTM")
    pb = load_close_panel(codes=union, field="pbMRQ")

    eval_dates = [pd.Timestamp(d) for d in sorted(mdf["date"].unique())
                  if pd.Timestamp(d) in close.index]

    factors = {
        "earnings_yield (1/PE)": fl.earnings_yield(pe),
        "book_yield (1/PB)": fl.book_yield(pb),
        "momentum 6-1m": fl.momentum(close, 120, 20),
        "low_vol 120d": fl.low_vol(close, 120),
        "reversal 1m": -fl.trailing_return(close, 20),
    }

    print(f"eval: {len(eval_dates)} month-end dates, PIT HS300 universe, "
          f"non-overlapping 1-month forward returns\n")
    print(f"  {'factor':<22} {'mean IC':>8} {'IC IR':>7} {'t-stat':>7} {'hit%':>6} {'Q5-Q1':>9}")
    for name, f in factors.items():
        ic = compute_ic(f, close, eval_dates, members_asof=asof)
        q = quantile_returns(f, close, eval_dates, n_q=5, members_asof=asof)
        spread = (q.iloc[-1] - q.iloc[0]) if len(q) == 5 else float("nan")
        print(f"  {name:<22} {ic.mean_ic:>+8.4f} {ic.ic_ir:>+7.3f} {ic.t_stat:>+7.2f} "
              f"{ic.hit_rate * 100:>5.0f}% {spread:>+9.2%}")

    print("\nRank IC = monthly cross-sectional Spearman(factor, next-month return) over PIT "
          "members. |t-stat|>~2 hints at a real (if modest) signal. A-share priors: value / "
          "low-vol / short-term reversal tend positive; 6-1m momentum is weak/negative.")


if __name__ == "__main__":
    main()
