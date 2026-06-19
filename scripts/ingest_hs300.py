"""Pull the full HS300 daily history (2015-01-01 .. 2025-12-31) into the parquet lake.

    conda activate hermes
    python scripts/ingest_hs300.py
"""
from hermes.data.ingest import ingest_hs300


def main() -> None:
    summary = ingest_hs300()
    errors = summary[summary["status"] != "ok"]
    if not errors.empty:
        print("\nsymbols with errors:")
        print(errors.to_string(index=False))


if __name__ == "__main__":
    main()
