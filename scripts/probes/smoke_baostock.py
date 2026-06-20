"""Smoke test: verify the BaoStock data link end to end.

    conda activate hermes
    python scripts/probes/smoke_baostock.py
"""
from hermes.data.sources import baostock_source as bss


def main() -> None:
    with bss.session():
        df = bss.daily_bars("sh.600000", "2024-01-02", "2024-01-10", adjustflag="2")
    print(f"rows: {len(df)}")
    print(df[["date", "code", "open", "close", "volume", "isST"]].to_string(index=False))


if __name__ == "__main__":
    main()
