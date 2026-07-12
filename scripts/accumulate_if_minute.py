"""Daily index-futures minute-bar accumulator -- run after market close on a trading day.

Sina only serves a shallow recent window (~1023 bars), so this UNIONs each day's pull into a
growing parquet lake (data/parquet/intraday/), building multi-year minute history forward. This is
the data-depth PREREQUISITE for any future intraday research (the first probe found no edge on
~1yr of data; see docs/intraday.md). Unlike US minute history (commercially backfillable later),
this window is gone if not accumulated. Tiny + idempotent: a few sub-second AKShare calls + small
atomic writes; safe to re-run or miss a day within the lookback window.

    python scripts/accumulate_if_minute.py        # all four CFFEX index futures, periods 1m + 5m
"""
from hermes.intraday.data import accumulate_futures_minute

SYMBOLS = [
    "IF0",   # CSI 300 main continuous -- accumulated since 2026-06-23
    "IC0",   # CSI 500 -- the standard mid/small-cap hedge leg (see index_hedge_study)
    "IH0",   # SSE 50 -- low standalone value (overlaps IF), kept for completeness at ~zero cost
    "IM0",   # CSI 1000 -- listed 2022, shortest history, so forward accumulation matters most
]            # IC0/IH0/IM0 added 2026-07-11; all verified served by the Sina endpoint


def main() -> None:
    for sym in SYMBOLS:
        totals = accumulate_futures_minute(sym)
        for period, rows in totals.items():
            print(f"{sym} {period}m: {rows} bars accumulated")
    print("OK")


if __name__ == "__main__":
    import sys
    try:
        main()
    except Exception as exc:                       # unattended: fail loud + nonzero exit
        print(f"ERROR: accumulate_if_minute failed: {exc}", file=sys.stderr)
        sys.exit(1)
