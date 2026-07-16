"""The conviction-sector creed: dip-add and hold, it always comes back. Issue #8.

A second friend's creed, pre-registered before this script existed: pick your sector, add on
every dip, hold -- it always comes back within tolerable time. Frozen design:

  indices    Shenwan sector closes (data/parquet/sw_indices.parquet); PRIMARY = the friend's
             named exposures (801081 semiconductors, 801740 defense, 801050 nonferrous);
             ROBUSTNESS = all 31 level-1 indices
  episode    first close <= 80% of the trailing 500d rolling peak; re-arms at 95% of the
             (then-current) peak
  ladder     equal thirds at -20% (trigger), -30%, -40% of the same peak; no exit
  verdict    CONFIRMED only if the +2%/yr dividend-credited ladder recovers its cost within
             500 trading days in >= 90% of decided level-1 episodes (undecided = fewer than
             500 subsequent days, excluded and counted)

    conda activate hermes
    python scripts/sector_creed_study.py
"""
from __future__ import annotations

import pandas as pd

from hermes.data.sw_indices import HS300, PRIMARY_CODES, SW1_CODES, load_sw_indices
from hermes.io import atomic_to_parquet
from hermes.paths import BACKTESTS_DIR, ensure_dirs
from hermes.research.backtest.drawdown_ladder import (DIV_CREDIT, episodes, ladder_outcome,
                                                      schedule_into)
from hermes.research.backtest.metrics import clustered_tstat

HORIZON = 500


def run_index(code: str, panel: pd.DataFrame) -> pd.DataFrame:
    close = panel[code].dropna()
    bench = panel[HS300]
    ep = episodes(close)
    rows = []
    for _, e in ep.iterrows():
        plain = ladder_outcome(close, int(e["iloc"]), e["peak"], HORIZON, div_credit=0.0)
        credited = ladder_outcome(close, int(e["iloc"]), e["peak"], HORIZON,
                                  div_credit=DIV_CREDIT)
        end_pos = min(int(e["iloc"]) + HORIZON, len(close) - 1)
        rows.append({"code": code, "date": e["date"], "decided": plain["n_days"] >= HORIZON,
                     "n_fills": plain["n_fills"],
                     "rec_plain": plain["recovered"], "rec_credit": credited["recovered"],
                     "rec_day": credited["recovered_day"],
                     "ladder": plain["terminal"], "lump": plain["lump_terminal"],
                     "hs300": schedule_into(bench, plain["fill_dates"],
                                            close.index[end_pos])})
    return pd.DataFrame(rows)


def _rate(df: pd.DataFrame, col: str) -> str:
    d = df[df["decided"]]
    return f"{d[col].mean():.1%} ({int(d[col].sum())}/{len(d)})" if len(d) else "n/a"


def main() -> None:
    ensure_dirs()
    panel = load_sw_indices()
    all_eps = pd.concat([run_index(c, panel) for c in SW1_CODES], ignore_index=True)
    prim = pd.concat([run_index(c, panel) for c in PRIMARY_CODES], ignore_index=True)

    decided = all_eps[all_eps["decided"]]
    quarters = pd.DatetimeIndex(decided["date"]).to_period("Q").nunique()
    print(f"episodes: {len(all_eps)} across {len(SW1_CODES)} level-1 indices "
          f"({len(decided)} decided, {len(all_eps) - len(decided)} undecided; "
          f"{quarters} distinct trigger quarters)")
    print(f"  recovery within {HORIZON}d -- price index: {_rate(all_eps, 'rec_plain')}; "
          f"+2%/yr credited (THE VERDICT): {_rate(all_eps, 'rec_credit')}")
    rd = decided.loc[decided["rec_day"] > 0, "rec_day"]
    print(f"  days to recovery (credited, recovered eps): "
          f"median {rd.median():.0f}, p75 {rd.quantile(0.75):.0f}, max {rd.max():.0f}")
    worst = decided.nsmallest(3, "ladder")[["code", "date", "ladder", "n_fills"]]
    print("  worst decided episodes (ladder terminal):")
    for _, w in worst.iterrows():
        print(f"    {w['code']} {w['date'].date()}: {w['ladder']:.2f}x ({w['n_fills']} fills)")

    print(f"\nfriend's named sectors: {len(prim)} episodes, "
          f"credited recovery {_rate(prim, 'rec_credit')}")
    for c in PRIMARY_CODES:
        sub = prim[prim["code"] == c]
        print(f"  {c}: {_rate(sub, 'rec_credit')} recovered; "
              f"mean ladder terminal {sub[sub['decided']]['ladder'].mean():.2f}x")

    lad_lump = (decided["ladder"] - decided["lump"])
    lad_hs = (decided["ladder"] - decided["hs300"])
    print(f"\nthe economics (decided episodes, {HORIZON}d terminal):")
    print(f"  ladder {decided['ladder'].mean():.3f}x vs lump {decided['lump'].mean():.3f}x "
          f"(diff {lad_lump.mean():+.3f}, quarterly-clustered t "
          f"{clustered_tstat(lad_lump, decided['date'], freq='Q'):.2f})")
    print(f"  ladder vs same schedule into HS300 {decided['hs300'].mean():.3f}x "
          f"(diff {lad_hs.mean():+.3f}, t {clustered_tstat(lad_hs, decided['date'], freq='Q'):.2f})")

    d = all_eps[all_eps["decided"]]
    rate = float(d["rec_credit"].mean())
    verdict = rate >= 0.90
    print(f"\nVERDICT: {'CONFIRMED' if verdict else 'REJECTED'} -- needs credited "
          f"{HORIZON}d cost-recovery >= 90% of decided episodes (got {rate:.1%})")

    atomic_to_parquet(all_eps, BACKTESTS_DIR / "sector_creed_episodes.parquet", index=False)
    print(f"saved -> {BACKTESTS_DIR / 'sector_creed_episodes.parquet'}")


if __name__ == "__main__":
    main()
