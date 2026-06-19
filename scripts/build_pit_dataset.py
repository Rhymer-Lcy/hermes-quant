"""Build the survivorship-free dataset in one shot:
  1. point-in-time HS300 membership (month-end snapshots, incl. removed names)
  2. daily price history for the full union (incl. delisted names)

    python scripts/build_pit_dataset.py
"""
from hermes.data.ingest import pull_universe
from hermes.data.membership import build_membership


def main() -> None:
    _, union = build_membership()
    print(f"pulling {len(union)} union names (incl. removed/delisted)...")
    summary = pull_universe(union)
    ok = int((summary["status"] == "ok").sum())
    print(f"pulled {ok}/{len(summary)} symbols")


if __name__ == "__main__":
    main()
