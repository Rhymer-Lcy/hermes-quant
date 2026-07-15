"""Inverted-pyramid scale-in vs lump-sum at the same entries. Issue #3.

The friend's execution rule at the issue #2 entries, pre-registered before this script
existed: a third at the entry close, a third at the first close 7.5% below it, a third at the
first close 15% below (the frozen midpoint of the stated "5% to 10% per step"); unfilled
tranches stay in cash at 0%. Horizon 250 of the name's own traded days; buys pay the
proportional buy rate; both arms mark to market at the horizon.

  events     every day a name NEWLY satisfies quality AND value (from quality_value_study)
  entry      the name's first traded close after the signal day (a suspension straddling the
             signal drops the event)
  verdict    CONFIRMED only if the pyramid arm's mean net account return exceeds lump-sum's
             with monthly-clustered t > 2; a Sharpe-only win is recorded, not a confirmation

Run quality_value_study.py first (it writes the event set).

    conda activate hermes
    python scripts/pyramid_entry_study.py
"""
from __future__ import annotations

import pandas as pd

from hermes.data.lake import load_close_panel
from hermes.io import atomic_to_parquet
from hermes.paths import BACKTESTS_DIR, ensure_dirs
from hermes.research.backtest.metrics import clustered_tstat, tstat
from hermes.research.backtest.rule_portfolio import proportional_rates
from hermes.research.backtest.scale_in import run_events

HORIZON = 250
ERAS = [("2015-01-01", "2019-12-31"), ("2020-01-01", "2026-12-31")]


def _line(tag: str, ev: pd.DataFrame) -> dict:
    d = ev["diff"]
    return {"tag": tag, "n": len(ev),
            "pyramid": float(ev["pyramid"].mean() - 1), "lump": float(ev["lump"].mean() - 1),
            "diff": float(d.mean()), "t_event": tstat(d),
            "t_month": clustered_tstat(d, ev["entry_date"], freq="M"),
            "hit": float((d > 0).mean())}


def _print(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'sample':>26} {'N':>5} {'pyramid':>8} {'lump':>8} {'diff':>8} {'t(event)':>9} "
          f"{'t(month)':>9} {'hit':>6}")
    for r in rows:
        print(f"  {r['tag']:>26} {r['n']:>5} {r['pyramid']:>+8.2%} {r['lump']:>+8.2%} "
              f"{r['diff']:>+8.2%} {r['t_event']:>9.2f} {r['t_month']:>9.2f} {r['hit']:>6.1%}")


def main() -> None:
    ensure_dirs()
    events = pd.read_parquet(BACKTESTS_DIR / "quality_value_events.parquet")
    events["date"] = pd.to_datetime(events["date"])
    close = load_close_panel(codes=sorted(events["code"].unique()), field="close")
    buy_rate, _ = proportional_rates()

    ev = run_events(events, close, horizon=HORIZON, buy_rate=buy_rate)
    print(f"events: {len(ev)} of {len(events)} signals resolved to an entry "
          f"({len(events) - len(ev)} dropped: suspension straddle or dead series)")
    print(f"tranches filled: 1 -> {(ev['filled'] == 1).mean():.1%}, "
          f"2 -> {(ev['filled'] == 2).mean():.1%}, 3 -> {(ev['filled'] == 3).mean():.1%}")

    # Cross-sectional dispersion read (the frozen "both legs' Sharpe over the pooled events"):
    for arm in ("pyramid", "lump"):
        x = ev[arm] - 1
        print(f"  {arm}: mean {x.mean():+.2%}, sd {x.std():.2%}, mean/sd {x.mean() / x.std():.3f}")

    rows = [_line("ALL events", ev),
            _line("ladder deployed (>=2 fills)", ev[ev["filled"] >= 2]),
            _line("never fell 7.5% (1 fill)", ev[ev["filled"] == 1])]
    for s, e in ERAS:
        cut = ev[(ev["entry_date"] >= s) & (ev["entry_date"] <= e)]
        rows.append(_line(f"{s[:4]}-{e[:4]}", cut))
    _print(rows, "INVERTED PYRAMID vs LUMP-SUM (net account return over 250 traded days):")

    head = rows[0]
    verdict = head["diff"] > 0 and head["t_month"] > 2
    print(f"\nVERDICT: {'CONFIRMED' if verdict else 'REJECTED'} -- needs pyramid > lump-sum "
          f"with monthly-clustered t > 2 (got diff {head['diff']:+.2%}, t {head['t_month']:.2f})")

    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "pyramid_entry_summary.parquet",
                      index=False)
    atomic_to_parquet(ev, BACKTESTS_DIR / "pyramid_entry_events.parquet", index=False)
    print(f"saved -> {BACKTESTS_DIR / 'pyramid_entry_summary.parquet'}")


if __name__ == "__main__":
    main()
