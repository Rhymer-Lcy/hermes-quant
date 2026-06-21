"""Paper-trading dry run: replay the DEPLOYED strategy (value + light 1-month-reversal tilt,
5/1, equal weight -- inverse-vol does not stack and the turnover buffer hurts, see
docs/multi_factor.md) as an idempotent EOD ledger at the roadmap capital tiers.

Two things this shows:
  1. ANTI-SKEW GATE: the ledger equity (reconstructed from the seed + folded fills) must
     equal the research engine's equity bar-for-bar. If it ever diverges, paper trading has
     drifted from research -- the run ASSERTS equality and fails loudly otherwise.
  2. SMALL-ACCOUNT FRICTION: avg_names_held vs the target 10. At ¥5k/¥10k the 100-share lot +
     ¥5 min commission make a 10-name book infeasible; the tier sweep makes that explicit
     (the same diagnostic the backtest engine exposes), which is the point of paper-trading
     across tiers before risking live money.

    python scripts/paper_demo.py
"""
import pandas as pd

from hermes.data.lake import load_close_panel
from hermes.data.membership import MEMBERSHIP_PARQUET, membership_lookup
from hermes.live.paper import ledger_equity, replay
from hermes.live.strategy import ALL_TIERS, DEPLOYED, deployed_signal

TIERS = ALL_TIERS
N_HOLD = DEPLOYED.n_hold


def main() -> None:
    mdf = pd.read_parquet(MEMBERSHIP_PARQUET)
    union = sorted(mdf["code"].unique())
    asof = membership_lookup(mdf)

    close = load_close_panel(codes=union, field="close")
    pe = load_close_panel(codes=union, field="peTTM")
    signal = deployed_signal(close, pe, asof)         # SAME spec research and live share

    print(f"deployed = value + 1m-reversal 5/1, top{N_HOLD} monthly, PIT HS300, A-share frictions\n")
    print(f"  {'tier':>9} {'final equity':>14} {'totRet':>8} {'maxDD':>8} {'avgNames':>9} {'trades':>7} {'parity':>8}")
    for cap in TIERS:
        ledger, res = replay(close, signal, cap, n_hold=N_HOLD, members_asof=asof)
        led_eq = ledger_equity(ledger)
        # Anti-skew gate: the ledger must reproduce the engine's equity bar-for-bar.
        eng_eq = res.equity
        aligned = led_eq.reindex(eng_eq.index)
        max_abs = float((aligned.values - eng_eq.values).__abs__().max())
        parity = "OK" if max_abs < 1e-6 else f"FAIL {max_abs:.2e}"
        assert max_abs < 1e-6, f"paper ledger diverged from research engine by {max_abs} at tier {cap}"
        print(f"  {cap:>9,} {led_eq.iloc[-1]:>14,.0f} {res.total_return:>+8.1%} "
              f"{res.max_drawdown:>8.1%} {res.avg_names_held:>9.2f} {len(res.trades):>7} {parity:>8}")

    print("\nParity holds at every tier: paper P&L is the research engine's P&L, reconstructed "
          "from the immutable seed -- no factor/sizing re-implementation, so no train/serve skew. "
          "avg_names_held collapses below the 10-name target at the small tiers (100-share lots + "
          "¥5 min commission) -- the small-account problem, made concrete before any live money. "
          "Next: a daily live_step that appends today's real EOD bar and extends the ledger.")


if __name__ == "__main__":
    main()
