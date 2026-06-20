"""Daily IF (沪深300 股指期货) minute-bar accumulator -- run after market close on a trading day.

Sina only serves a shallow recent window (~1023 bars), so this UNIONs each day's pull into a
growing parquet lake (data/parquet/intraday/), building multi-year minute history forward. This is
the data-depth PREREQUISITE for any future intraday-IF research (the first probe found no edge on
~1yr of data; see docs/intraday.md). Tiny + idempotent: a few sub-second AKShare calls + small
atomic writes; safe to re-run or miss a day within the lookback window.

    python scripts/accumulate_if_minute.py        # IF0, periods 1m + 5m
"""
from hermes.intraday.data import accumulate_futures_minute

SYMBOLS = ["IF0"]            # IF0 = 沪深300 main continuous; extensible to IC0/IH0 if breadth wanted


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
